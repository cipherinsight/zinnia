//! Subprocess driver for the Noir toolchain: `nargo execute` (compile and
//! solve the witness), `bb write_vk`, `bb prove` (the timed window, as the
//! `prove_core` window of the benchmark harness) and `bb verify`.
//!
//! Every `bb` call uses the same verifier target, `VERIFIER_TARGET`:
//! zero-knowledge UltraHonk with a Poseidon2 transcript, as the handwritten
//! Noir baselines and the benchmark harness (author's decision: Noir proofs
//! are zero-knowledge on both sides).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

/// `bb --verifier_target` for every call (write_vk, prove, verify).
pub const VERIFIER_TARGET: &str = "noir-recursive";

/// Package name written into the staged `Nargo.toml`; nargo names its
/// bytecode and witness files after it.
pub const PACKAGE_NAME: &str = "zinnia_circuit";

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

/// Toolchain lookup: the `ZINNIA_NARGO_BIN` / `ZINNIA_BB_BIN` overrides, then
/// the pinned installs (`~/.nargo/bin/nargo`, `~/.bb/bb`, the versions the
/// evaluation uses), then `PATH`. Preferring the pinned installs over `PATH`
/// keeps a stray binary on `PATH` from silently changing the toolchain.
impl Default for DriverPaths {
    fn default() -> Self {
        Self {
            nargo: locate("ZINNIA_NARGO_BIN", ".nargo/bin/nargo", "nargo"),
            bb: locate("ZINNIA_BB_BIN", ".bb/bb", "bb"),
        }
    }
}

fn locate(env: &str, home_rel: &str, name: &str) -> PathBuf {
    if let Ok(explicit) = std::env::var(env) {
        return PathBuf::from(explicit);
    }
    let pinned = home_dir().join(home_rel);
    if pinned.is_file() {
        return pinned;
    }
    which::which_in_path(name).unwrap_or(pinned)
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
    /// bb's public-inputs file: 32-byte big-endian field elements, the
    /// `pub` inputs of `main` first, then its return values.
    pub public_inputs_bytes: Vec<u8>,
    pub verify_ok: bool,
    /// Combined stdout+stderr tail from `bb verify` when it rejected the
    /// proof (`verify_ok == false`).
    pub verify_detail: Option<String>,
}

fn require(path: &Path, what: &str, env: &str) -> Result<(), NoirError> {
    if path.exists() {
        Ok(())
    } else {
        Err(NoirError::ToolchainMissing(format!(
            "{} not found at {} (override with {})",
            what,
            path.display(),
            env
        )))
    }
}

/// Run a toolchain command; a non-zero exit becomes `SubprocessFailed`.
fn run(step: &str, cmd: &mut Command) -> Result<Output, NoirError> {
    let out = cmd
        .output()
        .map_err(|e| NoirError::Io(format!("spawn {}: {}", step, e)))?;
    if out.status.success() {
        Ok(out)
    } else {
        Err(NoirError::SubprocessFailed {
            step: step.to_string(),
            detail: tail_combined(&out.stdout, &out.stderr, 1200),
        })
    }
}

fn read_file(path: &Path) -> Result<Vec<u8>, NoirError> {
    std::fs::read(path).map_err(|e| NoirError::Io(format!("read {}: {}", path.display(), e)))
}

/// `nargo execute --overwrite-return` in a staged package: compiles the
/// circuit, solves the witness (failing if any constraint is violated) and
/// returns the values `main` returned, as `0x`-prefixed field elements.
pub fn execute(pkg_dir: &Path, paths: &DriverPaths) -> Result<Vec<String>, NoirError> {
    require(&paths.nargo, "nargo", "ZINNIA_NARGO_BIN")?;
    run(
        "nargo execute",
        Command::new(&paths.nargo)
            .arg("execute")
            .arg("--overwrite-return")
            .current_dir(pkg_dir),
    )?;
    let prover_toml = std::fs::read_to_string(pkg_dir.join("Prover.toml"))
        .map_err(|e| NoirError::Io(format!("read Prover.toml: {}", e)))?;
    Ok(parse_return_entry(&prover_toml))
}

/// The `return = ...` entry nargo writes into Prover.toml: a single quoted
/// value or a list of them. A circuit without a return value has none.
fn parse_return_entry(prover_toml: &str) -> Vec<String> {
    let Some(line) = prover_toml
        .lines()
        .find(|l| l.trim_start().starts_with("return ="))
    else {
        return Vec::new();
    };
    line.split_once('=')
        .map(|(_, rhs)| rhs)
        .unwrap_or("")
        .split(|c| c == '[' || c == ']' || c == ',')
        .map(|tok| tok.trim().trim_matches('"').to_string())
        .filter(|tok| !tok.is_empty())
        .collect()
}

/// Full pipeline: execute, write the verification key, prove (timed), and
/// verify the proof against the public inputs bb wrote.
pub fn run_pipeline(pkg_dir: &Path, paths: &DriverPaths) -> Result<DriverOutput, NoirError> {
    require(&paths.bb, "bb", "ZINNIA_BB_BIN")?;
    execute(pkg_dir, paths)?;

    let target = pkg_dir.join("target");
    let bytecode = target.join(format!("{}.json", PACKAGE_NAME));
    let witness = target.join(format!("{}.gz", PACKAGE_NAME));

    // bb write_vk (excluded from the prove_seconds window).
    run(
        "bb write_vk",
        Command::new(&paths.bb)
            .arg("write_vk")
            .arg("-b")
            .arg(&bytecode)
            .arg("-o")
            .arg(&target)
            .args(["--scheme", "ultra_honk", "--verifier_target", VERIFIER_TARGET]),
    )?;

    // bb prove: the timed window (the harness times the same call after the
    // VK exists).
    let t = Instant::now();
    run(
        "bb prove",
        Command::new(&paths.bb)
            .arg("prove")
            .arg("-b")
            .arg(&bytecode)
            .arg("-w")
            .arg(&witness)
            .arg("-k")
            .arg(target.join("vk"))
            .arg("-o")
            .arg(&target)
            .args(["--scheme", "ultra_honk", "--verifier_target", VERIFIER_TARGET]),
    )?;
    let prove_seconds = t.elapsed().as_secs_f64();

    // bb always writes the public-inputs file (empty when there are none),
    // so a missing file is an error, not a reason to skip verification.
    let proof_bytes = read_file(&target.join("proof"))?;
    let vk_bytes = read_file(&target.join("vk"))?;
    let public_inputs_bytes = read_file(&target.join("public_inputs"))?;
    let (verify_ok, verify_detail) =
        match verify_proof(paths, &vk_bytes, &proof_bytes, &public_inputs_bytes)? {
            Ok(()) => (true, None),
            Err(detail) => (false, Some(detail)),
        };

    Ok(DriverOutput {
        prove_seconds,
        proof_bytes,
        vk_bytes,
        public_inputs_bytes,
        verify_ok,
        verify_detail,
    })
}

/// `bb verify` of `proof` against `vk` and `public_inputs` (raw file
/// contents). The outer error is a toolchain/IO failure; the inner `Err`
/// carries the verifier's diagnostics when it rejects the proof.
pub fn verify_proof(
    paths: &DriverPaths,
    vk: &[u8],
    proof: &[u8],
    public_inputs: &[u8],
) -> Result<Result<(), String>, NoirError> {
    require(&paths.bb, "bb", "ZINNIA_BB_BIN")?;
    let dir = scratch_dir("zinnia-noir-verify")?;
    let write = |name: &str, bytes: &[u8]| -> Result<PathBuf, NoirError> {
        let path = dir.join(name);
        std::fs::write(&path, bytes)
            .map_err(|e| NoirError::Io(format!("write {}: {}", path.display(), e)))?;
        Ok(path)
    };
    let vk_path = write("vk", vk)?;
    let proof_path = write("proof", proof)?;
    let pi_path = write("public_inputs", public_inputs)?;
    let out = Command::new(&paths.bb)
        .arg("verify")
        .arg("-k")
        .arg(&vk_path)
        .arg("-p")
        .arg(&proof_path)
        .arg("-i")
        .arg(&pi_path)
        .args(["--scheme", "ultra_honk", "--verifier_target", VERIFIER_TARGET])
        .output()
        .map_err(|e| NoirError::Io(format!("spawn bb verify: {}", e)));
    let _ = std::fs::remove_dir_all(&dir);
    let out = out?;
    if out.status.success() {
        Ok(Ok(()))
    } else {
        Ok(Err(tail_combined(&out.stdout, &out.stderr, 1200)))
    }
}

/// A fresh directory under the system temp dir (PID + nanos, no uuid dep).
pub fn scratch_dir(prefix: &str) -> Result<PathBuf, NoirError> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("{}-{}-{}", prefix, std::process::id(), nanos));
    std::fs::create_dir_all(&dir)
        .map_err(|e| NoirError::Io(format!("mkdir {}: {}", dir.display(), e)))?;
    Ok(dir)
}

fn tail_combined(stdout: &[u8], stderr: &[u8], max: usize) -> String {
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(stdout));
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(stderr));
    if combined.len() > max {
        let mut start = combined.len() - max;
        while !combined.is_char_boundary(start) {
            start += 1;
        }
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

#[cfg(test)]
mod tests {
    use super::parse_return_entry;

    #[test]
    fn return_entry_list_and_scalar() {
        let toml = "return = [\"0x01\", \"0x02\"]\nx = \"0x05\"\n";
        assert_eq!(parse_return_entry(toml), vec!["0x01", "0x02"]);
        assert_eq!(parse_return_entry("return = \"0x07\"\n"), vec!["0x07"]);
        assert!(parse_return_entry("x = \"1\"\n").is_empty());
    }
}
