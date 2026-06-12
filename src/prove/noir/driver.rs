//! Subprocess driver that runs `nargo execute` then `bb prove --scheme
//! ultra_honk`. VK generation is split off into its own `bb write_vk`
//! call so the timed window around `bb prove` matches what
//! `benchmarking/run_noir_baselines.py` measures.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug)]
pub enum NoirError {
    ToolchainMissing(String),
    SubprocessFailed { step: String, detail: String },
    Io(String),
}

pub struct DriverPaths {
    pub nargo: PathBuf,
    pub bb: PathBuf,
}

impl Default for DriverPaths {
    fn default() -> Self {
        let nargo = std::env::var("ZINNIA_NARGO_BIN")
            .map(PathBuf::from)
            .ok()
            .or_else(|| which::which_in_path("nargo"))
            .unwrap_or_else(|| home_dir().join(".nargo/bin/nargo"));
        let bb = std::env::var("ZINNIA_BB_BIN")
            .map(PathBuf::from)
            .ok()
            .or_else(|| which::which_in_path("bb"))
            .unwrap_or_else(|| home_dir().join(".bb/bb"));
        Self { nargo, bb }
    }
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"))
}

#[derive(Debug)]
pub struct DriverOutput {
    pub prove_seconds: f64,
    pub proof_bytes: Vec<u8>,
    pub vk_bytes: Vec<u8>,
    pub public_inputs_bytes: Vec<u8>,
    pub verify_ok: bool,
    /// Combined stdout+stderr tail from `bb verify` when it rejected the
    /// proof (`verify_ok == false`). `None` on the success path or when no
    /// `public_inputs` file was emitted (verify skipped). Surfaced by
    /// `prove()` in the `ProvingFailed` error so a bb-rejected proof carries
    /// the verifier's own diagnostics.
    pub verify_detail: Option<String>,
}

pub fn run_pipeline(pkg_dir: &Path, paths: &DriverPaths) -> Result<DriverOutput, NoirError> {
    if !paths.nargo.exists() {
        return Err(NoirError::ToolchainMissing(format!(
            "nargo not found at {} (override with ZINNIA_NARGO_BIN)",
            paths.nargo.display()
        )));
    }
    if !paths.bb.exists() {
        return Err(NoirError::ToolchainMissing(format!(
            "bb not found at {} (override with ZINNIA_BB_BIN)",
            paths.bb.display()
        )));
    }

    // 1. nargo execute (witness generation, also produces ACIR bytecode).
    let out = Command::new(&paths.nargo)
        .arg("execute")
        .current_dir(pkg_dir)
        .output()
        .map_err(|e| NoirError::Io(format!("spawn nargo: {}", e)))?;
    if !out.status.success() {
        return Err(NoirError::SubprocessFailed {
            step: "nargo execute".into(),
            detail: tail_combined(&out.stdout, &out.stderr, 1200),
        });
    }

    // The bytecode lives at target/zinnia_circuit.json and the witness
    // at target/zinnia_circuit.gz (package name = "zinnia_circuit").
    let target = pkg_dir.join("target");
    let bytecode = target.join("zinnia_circuit.json");
    let witness = target.join("zinnia_circuit.gz");
    if !bytecode.exists() || !witness.exists() {
        return Err(NoirError::SubprocessFailed {
            step: "nargo execute".into(),
            detail: format!(
                "expected {} and {} after nargo execute",
                bytecode.display(),
                witness.display()
            ),
        });
    }

    // 2. bb write_vk (excluded from the prove_seconds window).
    let out = Command::new(&paths.bb)
        .arg("write_vk")
        .arg("-b")
        .arg(&bytecode)
        .arg("-o")
        .arg(&target)
        .arg("--scheme")
        .arg("ultra_honk")
        .arg("--verifier_target")
        .arg("noir-recursive-no-zk")
        .output()
        .map_err(|e| NoirError::Io(format!("spawn bb write_vk: {}", e)))?;
    if !out.status.success() {
        return Err(NoirError::SubprocessFailed {
            step: "bb write_vk".into(),
            detail: tail_combined(&out.stdout, &out.stderr, 1200),
        });
    }
    let vk_path = target.join("vk");

    // 3. bb prove (timed window — matches run_noir_baselines.py
    // methodology, which measures wall-clock around `bb prove` after
    // the VK was already generated).
    let t = Instant::now();
    let out = Command::new(&paths.bb)
        .arg("prove")
        .arg("-b")
        .arg(&bytecode)
        .arg("-w")
        .arg(&witness)
        .arg("-k")
        .arg(&vk_path)
        .arg("-o")
        .arg(&target)
        .arg("--scheme")
        .arg("ultra_honk")
        .arg("--verifier_target")
        .arg("noir-recursive-no-zk")
        .output()
        .map_err(|e| NoirError::Io(format!("spawn bb prove: {}", e)))?;
    let prove_seconds = t.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(NoirError::SubprocessFailed {
            step: "bb prove".into(),
            detail: tail_combined(&out.stdout, &out.stderr, 1200),
        });
    }
    let proof_path = target.join("proof");

    // 4. bb verify.
    let pi_path = target.join("public_inputs");
    let (verify_ok, verify_detail) = if pi_path.exists() {
        let out = Command::new(&paths.bb)
            .arg("verify")
            .arg("-k")
            .arg(&vk_path)
            .arg("-p")
            .arg(&proof_path)
            .arg("-i")
            .arg(&pi_path)
            .arg("--scheme")
            .arg("ultra_honk")
            .arg("--verifier_target")
            .arg("noir-recursive-no-zk")
            .output()
            .map_err(|e| NoirError::Io(format!("spawn bb verify: {}", e)))?;
        if out.status.success() {
            (true, None)
        } else {
            // Capture the verifier's own diagnostics so prove() can ship
            // them in the ProvingFailed error rather than silently emitting
            // a bb-rejected proof as a valid artifact.
            (false, Some(tail_combined(&out.stdout, &out.stderr, 1200)))
        }
    } else {
        // No public_inputs file emitted (kernel without exposes) — treat
        // the bb prove return code as sufficient.
        (true, None)
    };

    let proof_bytes = std::fs::read(&proof_path).unwrap_or_default();
    let vk_bytes = std::fs::read(&vk_path).unwrap_or_default();
    let public_inputs_bytes = std::fs::read(&pi_path).unwrap_or_default();

    Ok(DriverOutput {
        prove_seconds,
        proof_bytes,
        vk_bytes,
        public_inputs_bytes,
        verify_ok,
        verify_detail,
    })
}

fn tail_combined(stdout: &[u8], stderr: &[u8], max: usize) -> String {
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(stdout));
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(stderr));
    if combined.len() > max {
        let start = combined.len() - max;
        format!("...{}", &combined[start..])
    } else {
        combined
    }
}

mod which {
    use std::path::PathBuf;

    pub fn which_in_path(prog: &str) -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        for dir in std::env::split_paths(&path) {
            let p = dir.join(prog);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }
}
