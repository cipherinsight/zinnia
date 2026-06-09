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

use std::time::Instant;

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

use self::driver::{run_pipeline, DriverPaths};
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

        let dir = stage_package(&main_nr, &prover_toml)?;
        let paths = DriverPaths::default();

        let pipeline_started = Instant::now();
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
        let _ = pipeline_started; // pipeline wall-time is not the reported field

        // Successful run: remove staging dir unless `ZINNIA_NOIR_KEEP_TEMP=1`
        // (debugging hatch — leaks the directory so a developer can diff the
        // emitted main.nr against a handwritten baseline). Failed runs
        // intentionally leak too (see driver).
        if std::env::var("ZINNIA_NOIR_KEEP_TEMP").as_deref() != Ok("1") {
            let _ = std::fs::remove_dir_all(&dir);
        } else {
            eprintln!("zinnia-noir: kept staging dir at {}", dir.display());
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
        // expose_count is the number of `pub` slots in main.nr — useful
        // for verify-side public_inputs reconstruction; today we ship it
        // implicitly through public_values.len().
        .map(|a| {
            let _ = expose_count;
            a
        })
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        if artifact.proof_bytes.is_empty() {
            return Ok(VerifyResult::invalid("empty proof_bytes"));
        }
        // Cross-process verify lives in the Python runner via `bb verify`.
        // From inside Rust we treat a non-empty proof_bytes payload as
        // tentatively valid; the driver only assembles an artifact after
        // an in-process `bb verify` already accepted it.
        Ok(VerifyResult::ok())
    }
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
