//! Source-to-source Noir proving backend.
//!
//! Emits a self-contained Noir package (`Nargo.toml`, `src/main.nr`,
//! `Prover.toml`, vendored `noir_fpc/` library) from a Zinnia IR graph
//! plus its resolved witness, then shells out to `nargo execute` +
//! `bb prove --scheme ultra_honk` to produce a proof. The `prove_seconds`
//! field on the resulting [`ProofArtifact`] times only the `bb prove`
//! call (after `bb write_vk` runs separately), matching the
//! apples-to-apples timing convention used by the halo2 backend.
//!
//! ## Public values and verification
//!
//! The artifact's `public_values` are the public inputs `bb` proved, read
//! back from its public-inputs file (`pub` inputs first, then the returned
//! outputs), and [`NoirProverBackend::verify`] runs `bb verify` against
//! them. [`NoirProverBackend::execute`] runs the circuit without proving,
//! for the differential test against the mock backend.
//!
//! ## Vendoring the fp library
//!
//! The Q32 fixed-point library at `benchmarking/noir_fpc` is the single
//! source of truth for both the handwritten baselines and the
//! auto-emitted circuits. We pull it in at compile time via `include_str!`
//! so a fresh build always picks up the latest library text without a
//! sync step.

pub mod bound_analysis;
pub mod driver;
pub mod emit;
pub mod inputs;
#[cfg(test)]
mod soundness_tests;

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

use self::driver::{run_pipeline, DriverOutput, DriverPaths};
use self::emit::emit_circuit;

const NOIR_FPC_NARGO_TOML: &str = include_str!("../../../benchmarking/noir_fpc/Nargo.toml");
const NOIR_FPC_LIB_NR: &str = include_str!("../../../benchmarking/noir_fpc/src/lib.nr");

pub const NOIR_BACKEND_NAME: &str = "noir-ultrahonk";
pub const NOIR_BACKEND_FORMAT: &str = "noir-ultrahonk-v1";

pub struct NoirProverBackend;

impl NoirProverBackend {
    /// Emit and stage the circuit, then `nargo execute` it without proving:
    /// the values `main` returns (the exposed outputs, IR order) as
    /// `0x`-prefixed field elements. Fails if any constraint is violated.
    /// Used to compare the Noir semantics with the mock backend's.
    pub fn execute(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> Result<Vec<String>, ProvingError> {
        let emitted = emit_circuit(ir, witness, params)?;
        let dir = stage_package(&emitted.main_nr, &emitted.prover_toml)?;
        let result = driver::execute(&dir, &DriverPaths::default());
        cleanup_staging_dir(&dir, result.is_ok());
        let returned = result.map_err(noir_error)?;
        // The emitter may append one synthetic return to an otherwise empty
        // circuit; only the exposed outputs are reported.
        Ok(returned.into_iter().take(emitted.return_count).collect())
    }
}

impl ProverBackend for NoirProverBackend {
    fn name(&self) -> &'static str {
        NOIR_BACKEND_NAME
    }

    fn estimate_params(&self, _ir: &IRGraph) -> Result<ProvingParams, ProvingError> {
        // Noir+UltraHonk doesn't take a `k` from us — `bb` derives the
        // circuit size from the bytecode itself. We echo the default
        // ProvingParams so callers get a consistent precision_bits value.
        Ok(ProvingParams::default())
    }

    fn prove(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> Result<ProofArtifact, ProvingError> {
        let emitted = emit_circuit(ir, witness, params)?;
        let dir = stage_package(&emitted.main_nr, &emitted.prover_toml)?;
        let result = run_pipeline(&dir, &DriverPaths::default());
        cleanup_staging_dir(&dir, result.is_ok());
        assemble_artifact(result.map_err(noir_error)?)
    }

    /// Runs `bb verify` on the artifact's proof, verification key and public
    /// values. The public values are part of the statement: changing any of
    /// them makes verification fail.
    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        let vk = hex_decode(&artifact.vk_bytes)?;
        let proof = hex_decode(&artifact.proof_bytes)?;
        if proof.is_empty() {
            return Ok(VerifyResult::invalid("empty proof_bytes"));
        }
        let public_inputs = encode_public_inputs(&artifact.public_values)?;
        match driver::verify_proof(&DriverPaths::default(), &vk, &proof, &public_inputs)
            .map_err(noir_error)?
        {
            Ok(()) => Ok(VerifyResult::ok()),
            Err(detail) => Ok(VerifyResult::invalid(format!("bb verify rejected the proof: {}", detail))),
        }
    }
}

fn noir_error(e: driver::NoirError) -> ProvingError {
    match e {
        driver::NoirError::ToolchainMissing(detail) => {
            ProvingError::other(format!("Noir toolchain missing: {}", detail))
        }
        driver::NoirError::SubprocessFailed { step, detail } => ProvingError::ProvingFailed {
            detail: format!("Noir pipeline `{}` failed: {}", step, detail),
        },
        driver::NoirError::Io(detail) => ProvingError::other(format!("Noir IO error: {}", detail)),
    }
}

/// Remove a staging directory after a successful run, unless
/// `ZINNIA_NOIR_KEEP_TEMP=1` (debugging hatch: keeps the emitted package).
/// Failed runs keep it so the failure can be reproduced.
fn cleanup_staging_dir(dir: &std::path::Path, succeeded: bool) {
    if succeeded && std::env::var("ZINNIA_NOIR_KEEP_TEMP").as_deref() != Ok("1") {
        let _ = std::fs::remove_dir_all(dir);
    } else {
        eprintln!("zinnia-noir: kept staging dir at {}", dir.display());
    }
}

/// Turn a successful [`DriverOutput`] into a [`ProofArtifact`], gating on the
/// driver's in-process `bb verify` result. The artifact's `public_values`
/// are the public inputs bb proved (decoded from its public-inputs file:
/// `pub` inputs of `main` first, then the returned outputs), as
/// `0x`-prefixed 32-byte field elements, not values recomputed by the
/// emitter.
fn assemble_artifact(out: DriverOutput) -> Result<ProofArtifact, ProvingError> {
    if !out.verify_ok {
        let detail = out
            .verify_detail
            .unwrap_or_else(|| "bb verify rejected the proof (no diagnostics captured)".to_string());
        return Err(ProvingError::ProvingFailed {
            detail: format!("Noir `bb verify` rejected the generated proof: {}", detail),
        });
    }

    Ok(ProofArtifact {
        backend: NOIR_BACKEND_NAME.to_string(),
        format: NOIR_BACKEND_FORMAT.to_string(),
        vk_bytes: hex_encode(&out.vk_bytes),
        ir_hash: String::new(),
        proof_bytes: hex_encode(&out.proof_bytes),
        public_values: decode_public_inputs(&out.public_inputs_bytes)?,
        k: 0,
        prove_seconds: out.prove_seconds,
    })
}

/// bb's public-inputs file -> one `0x`-prefixed hex string per 32-byte
/// big-endian field element.
fn decode_public_inputs(bytes: &[u8]) -> Result<Vec<String>, ProvingError> {
    if bytes.len() % 32 != 0 {
        return Err(ProvingError::other(format!(
            "bb public-inputs file has {} bytes, not a multiple of 32",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks(32)
        .map(|chunk| format!("0x{}", hex_encode(chunk)))
        .collect())
}

/// Inverse of [`decode_public_inputs`].
fn encode_public_inputs(values: &[String]) -> Result<Vec<u8>, ProvingError> {
    let mut out = Vec::with_capacity(values.len() * 32);
    for v in values {
        let digits = v.strip_prefix("0x").unwrap_or(v);
        let bytes = hex_decode(digits)?;
        if bytes.len() != 32 {
            return Err(ProvingError::other(format!(
                "public value {} is not a 32-byte field element",
                v
            )));
        }
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// Materialize a temporary Noir package directory containing the emitted
/// `main.nr` + `Prover.toml`, alongside a vendored copy of `noir_fpc`.
fn stage_package(main_nr: &str, prover_toml: &str) -> Result<std::path::PathBuf, ProvingError> {
    let dir = driver::scratch_dir("zinnia-noir").map_err(noir_error)?;
    let lib_dir = dir.join("noir_fpc").join("src");
    let src_dir = dir.join("src");
    for d in [&src_dir, &lib_dir] {
        std::fs::create_dir_all(d).map_err(|e| ProvingError::other(format!("mkdir: {}", e)))?;
    }
    let write = |path: std::path::PathBuf, content: &str| -> Result<(), ProvingError> {
        std::fs::write(&path, content)
            .map_err(|e| ProvingError::other(format!("write {}: {}", path.display(), e)))
    };
    write(
        dir.join("Nargo.toml"),
        // The package name has to be a valid Noir identifier — no dashes.
        &format!(
            "[package]\nname = \"{}\"\ntype = \"bin\"\nauthors = [\"\"]\n\n\
             [dependencies]\nnoir_fpc = {{ path = \"./noir_fpc\" }}\n",
            driver::PACKAGE_NAME
        ),
    )?;
    write(src_dir.join("main.nr"), main_nr)?;
    write(dir.join("Prover.toml"), prover_toml)?;
    write(dir.join("noir_fpc").join("Nargo.toml"), NOIR_FPC_NARGO_TOML)?;
    write(lib_dir.join("lib.nr"), NOIR_FPC_LIB_NR)?;
    Ok(dir)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, ProvingError> {
    if s.len() % 2 != 0 {
        return Err(ProvingError::other("hex string has odd length"));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|_| ProvingError::other(format!("invalid hex at offset {}", i)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field_bytes(v: u8) -> Vec<u8> {
        let mut b = vec![0u8; 32];
        b[31] = v;
        b
    }

    fn driver_output(verify_ok: bool, verify_detail: Option<String>) -> DriverOutput {
        DriverOutput {
            prove_seconds: 1.5,
            proof_bytes: vec![0xde, 0xad, 0xbe, 0xef],
            vk_bytes: vec![0x01, 0x02],
            public_inputs_bytes: [field_bytes(7), field_bytes(42)].concat(),
            verify_ok,
            verify_detail,
        }
    }

    #[test]
    fn assemble_artifact_propagates_verify_failure() {
        // A run where bb verify rejected the proof must surface as
        // ProvingFailed carrying the verifier's diagnostics, not an artifact.
        let out = driver_output(false, Some("bb: proof verification failed\n".to_string()));
        let err = assemble_artifact(out).expect_err("verify_ok == false must be an Err");
        match err {
            ProvingError::ProvingFailed { detail } => {
                assert!(detail.contains("bb verify") && detail.contains("rejected"), "{detail}");
                assert!(detail.contains("proof verification failed"), "{detail}");
            }
            other => panic!("expected ProvingFailed, got {other:?}"),
        }
        let out = driver_output(false, None);
        assert!(matches!(assemble_artifact(out), Err(ProvingError::ProvingFailed { .. })));
    }

    #[test]
    fn assemble_artifact_takes_public_values_from_bb() {
        // The public values are bb's public inputs, not emitter-side values.
        let artifact = assemble_artifact(driver_output(true, None)).expect("artifact");
        assert_eq!(artifact.backend, NOIR_BACKEND_NAME);
        assert_eq!(artifact.proof_bytes, "deadbeef");
        assert_eq!(artifact.vk_bytes, "0102");
        assert_eq!(artifact.public_values.len(), 2);
        assert!(artifact.public_values[0].ends_with("07"));
        assert!(artifact.public_values[1].ends_with("2a"));
        assert_eq!(artifact.prove_seconds, 1.5);
    }

    #[test]
    fn public_inputs_round_trip_and_reject_ragged_files() {
        let bytes = [field_bytes(1), field_bytes(255)].concat();
        let values = decode_public_inputs(&bytes).unwrap();
        assert_eq!(encode_public_inputs(&values).unwrap(), bytes);
        assert!(decode_public_inputs(&[0u8; 33]).is_err());
        assert!(encode_public_inputs(&["0x01".to_string()]).is_err());
    }
}
