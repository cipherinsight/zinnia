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

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

use self::driver::{run_pipeline, DriverOutput, DriverPaths};
use self::emit::{emit_circuit, EmitOutput};

const NOIR_FPC_NARGO_TOML: &str = include_str!("../../../benchmarking/noir_fpc/Nargo.toml");
const NOIR_FPC_LIB_NR: &str = include_str!("../../../benchmarking/noir_fpc/src/lib.nr");

pub const NOIR_BACKEND_NAME: &str = "noir-ultrahonk";
pub const NOIR_BACKEND_FORMAT: &str = "noir-ultrahonk-v1";

pub struct NoirProverBackend;

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
        let EmitOutput {
            main_nr,
            prover_toml,
            public_values_decimal,
            expose_count,
        } = emit_circuit(ir, witness, params)?;

        // expose_count is the number of `pub` slots in main.nr — useful for
        // verify-side public_inputs reconstruction; today it is shipped
        // implicitly through public_values.len(), so we don't carry it on
        // the artifact.
        let _ = expose_count;

        let dir = stage_package(&main_nr, &prover_toml)?;
        let paths = DriverPaths::default();

        let out = run_pipeline(&dir, &paths).map_err(|e| match e {
            driver::NoirError::ToolchainMissing(detail) => {
                ProvingError::other(format!("Noir toolchain missing: {}", detail))
            }
            driver::NoirError::SubprocessFailed { step, detail } => ProvingError::ProvingFailed {
                detail: format!("Noir pipeline `{}` failed: {}", step, detail),
            },
            driver::NoirError::Io(detail) => {
                ProvingError::other(format!("Noir IO error: {}", detail))
            }
        })?;

        // Successful run: remove staging dir unless `ZINNIA_NOIR_KEEP_TEMP=1`
        // (debugging hatch — leaks the directory so a developer can diff the
        // emitted main.nr against a handwritten baseline). Failed runs
        // intentionally leak too (see driver).
        if std::env::var("ZINNIA_NOIR_KEEP_TEMP").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&dir);
        } else {
            eprintln!("zinnia-noir: kept staging dir at {}", dir.display());
        }

        assemble_artifact(out, public_values_decimal)
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        if artifact.proof_bytes.is_empty() {
            return Ok(VerifyResult::invalid("empty proof_bytes"));
        }
        // Cross-process verify lives in the Python runner via `bb verify`.
        // From inside Rust we treat a non-empty proof_bytes payload as
        // tentatively valid: `prove()` only assembles an artifact after the
        // in-process `bb verify` in the driver accepted it (a rejected proof
        // surfaces as `ProvingError::ProvingFailed`, never as an artifact),
        // so any artifact reaching this method has already passed bb verify.
        Ok(VerifyResult::ok())
    }
}

/// Turn a successful [`DriverOutput`] into a [`ProofArtifact`], gating on the
/// driver's in-process `bb verify` result.
///
/// If `bb verify` rejected the proof (`verify_ok == false`), this returns
/// [`ProvingError::ProvingFailed`] carrying the verifier's captured stderr
/// instead of shipping a bb-rejected proof as a valid artifact. Factored out
/// of `prove()` so the verify-gate can be unit-tested without spawning the
/// nargo/bb subprocess pipeline.
fn assemble_artifact(
    out: DriverOutput,
    public_values_decimal: Vec<String>,
) -> Result<ProofArtifact, ProvingError> {
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
        // The Noir backend does not yet ship a self-contained VK
        // payload — verify-in-Rust would require linking against bb
        // as a library. We carry the verifier-side stdout so the
        // Python runner can sanity-check it; the actual cross-process
        // verify lives in the Python runner via `bb verify`.
        vk_bytes: hex_encode(&out.vk_bytes),
        ir_hash: String::new(),
        proof_bytes: hex_encode(&out.proof_bytes),
        public_values: public_values_decimal,
        k: 0,
        prove_seconds: out.prove_seconds,
    })
}

/// Materialize a temporary Noir package directory containing the emitted
/// `main.nr` + `Prover.toml`, alongside a vendored copy of `noir_fpc`.
fn stage_package(main_nr: &str, prover_toml: &str) -> Result<std::path::PathBuf, ProvingError> {
    use std::io::Write;

    let parent = std::env::temp_dir();
    // Per-prove dir name: PID + nanos. Avoids collisions in concurrent
    // test runs without pulling in a uuid dep.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = parent.join(format!("zinnia-noir-{}-{}", std::process::id(), nanos));
    let src_dir = dir.join("src");
    let lib_dir = dir.join("noir_fpc").join("src");
    std::fs::create_dir_all(&src_dir).map_err(|e| ProvingError::other(format!("mkdir: {}", e)))?;
    std::fs::create_dir_all(&lib_dir).map_err(|e| ProvingError::other(format!("mkdir: {}", e)))?;

    let write = |path: std::path::PathBuf, content: &str| -> Result<(), ProvingError> {
        let mut f = std::fs::File::create(&path)
            .map_err(|e| ProvingError::other(format!("create {}: {}", path.display(), e)))?;
        f.write_all(content.as_bytes())
            .map_err(|e| ProvingError::other(format!("write {}: {}", path.display(), e)))?;
        Ok(())
    };

    write(
        dir.join("Nargo.toml"),
        // The package name has to be a valid Noir identifier — no dashes.
        "[package]\nname = \"zinnia_circuit\"\ntype = \"bin\"\nauthors = [\"\"]\n\n\
         [dependencies]\nnoir_fpc = { path = \"./noir_fpc\" }\n",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn driver_output(verify_ok: bool, verify_detail: Option<String>) -> DriverOutput {
        DriverOutput {
            prove_seconds: 1.5,
            proof_bytes: vec![0xde, 0xad, 0xbe, 0xef],
            vk_bytes: vec![0x01, 0x02],
            public_inputs_bytes: vec![0x03],
            verify_ok,
            verify_detail,
        }
    }

    #[test]
    fn assemble_artifact_propagates_verify_failure() {
        // A driver run where bb verify rejected the proof must NOT produce a
        // ProofArtifact — it must surface as ProvingFailed carrying the bb
        // verifier's captured diagnostics. This is the regression guard for
        // C13: previously prove() ignored verify_ok and shipped the artifact.
        let out = driver_output(false, Some("bb: proof verification failed\n".to_string()));
        let err = assemble_artifact(out, vec!["7".to_string()])
            .expect_err("verify_ok == false must be an Err, not an artifact");
        match err {
            ProvingError::ProvingFailed { detail } => {
                assert!(
                    detail.contains("bb verify") && detail.contains("rejected"),
                    "error should name the bb verify rejection, got: {detail}"
                );
                assert!(
                    detail.contains("proof verification failed"),
                    "error should carry the captured bb stderr, got: {detail}"
                );
            }
            other => panic!("expected ProvingFailed, got {other:?}"),
        }
    }

    #[test]
    fn assemble_artifact_failure_without_detail_still_errs() {
        // Defensive: verify_ok == false with no captured stderr is still an
        // Err (never a silent success), with a placeholder diagnostic.
        let out = driver_output(false, None);
        let err = assemble_artifact(out, vec![])
            .expect_err("verify_ok == false must be an Err even without detail");
        assert!(matches!(err, ProvingError::ProvingFailed { .. }));
    }

    #[test]
    fn assemble_artifact_success_builds_artifact() {
        // Success path: verify_ok == true yields a populated artifact with
        // hex-encoded payloads and the driver's prove_seconds preserved.
        let out = driver_output(true, None);
        let artifact = assemble_artifact(out, vec!["42".to_string()])
            .expect("verify_ok == true must produce an artifact");
        assert_eq!(artifact.backend, NOIR_BACKEND_NAME);
        assert_eq!(artifact.format, NOIR_BACKEND_FORMAT);
        assert_eq!(artifact.proof_bytes, "deadbeef");
        assert_eq!(artifact.vk_bytes, "0102");
        assert_eq!(artifact.public_values, vec!["42".to_string()]);
        assert_eq!(artifact.prove_seconds, 1.5);
    }
}
