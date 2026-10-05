//! SP1 host for the DS-1000 baselines (SP1 6.8.1, CPU prover).
//!
//! Modes (see `zkvm_kernels::host` for the program contract):
//! * `setup`: compute the proving and verifying keys of the case's guest and
//!   store them in the work directory;
//! * `execute`: run the guest in SP1's executor; public values and cycles;
//! * `prove`: execute (timed on its own), then prove in the chosen proof mode
//!   and store the proof. SP1's prover re-executes the guest internally, so
//!   `timings_s.prove` includes that second execution;
//! * `verify`: verify the stored proof against the stored verifying key.
//!
//! Every mode first creates the prover client, timed as
//! `timings_s.client_init` (16-21 s for the CPU prover, which loads its
//! recursion artefacts). It is one-time setup, like Zinnia's key generation:
//! reported with setup, never part of `timings_s.prove`.
//!
//! Proof modes: `core` (one STARK per execution shard), `compressed`
//! (recursively compressed to one STARK), `groth16` (the compressed proof
//! wrapped in a Groth16 SNARK over BN254, proved with gnark linked in-process).

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use zkvm_kernels::{
    host::{run_host, tamper_byte, CommonArgs, Mode, Report},
    words_as_le_bytes, words_from_le_bytes,
};
use serde::{de::DeserializeOwned, Serialize};
use sp1_sdk::{
    blocking::{CpuProver, LightProver, ProveRequest, Prover},
    Elf, HashableKey, ProvingKey, SP1ProofMode, SP1ProofWithPublicValues, SP1ProvingKey,
    SP1PublicValues, SP1Stdin, SP1VerifyingKey,
};

include!(concat!(env!("OUT_DIR"), "/guest_elfs.rs"));

const SDK_VERSION: &str = "6.8.1";

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, value_enum, default_value = "core")]
    proof_mode: ProofMode,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProofMode {
    Core,
    Compressed,
    Groth16,
}

impl ProofMode {
    fn sdk_mode(self) -> SP1ProofMode {
        match self {
            ProofMode::Core => SP1ProofMode::Core,
            ProofMode::Compressed => SP1ProofMode::Compressed,
            ProofMode::Groth16 => SP1ProofMode::Groth16,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ProofMode::Core => "core",
            ProofMode::Compressed => "compressed",
            ProofMode::Groth16 => "groth16",
        }
    }
}

/// The files a case keeps in its work directory.
struct WorkFiles {
    proving_key: PathBuf,
    verifying_key: PathBuf,
    dir: PathBuf,
}

impl WorkFiles {
    fn new(args: &CommonArgs) -> Result<WorkFiles> {
        let dir = args.work_dir(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())?;
        Ok(WorkFiles {
            proving_key: dir.join("pk.bin"),
            verifying_key: dir.join("vk.bin"),
            dir,
        })
    }

    fn proof(&self, mode: ProofMode, input: &str) -> PathBuf {
        self.dir.join(format!("proof-{}-{input}.bin", mode.name()))
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    assert_cpu_prover();
    run_host("sp1", &cli.common, |report| {
        report.set("sdk_version", SDK_VERSION);
        let case = cli.common.case()?;
        let elf = load_elf(case)?;
        let work = WorkFiles::new(&cli.common)?;
        match cli.common.mode {
            Mode::Setup => setup(report, elf, &work),
            Mode::Execute => execute(report, elf, &cli.common),
            Mode::Prove => prove(report, elf, &cli, &work),
            Mode::Verify => verify(report, &cli, &work),
        }
    })
}

/// The baselines run on the CPU prover only.
fn assert_cpu_prover() {
    match std::env::var("SP1_PROVER") {
        Ok(prover) if prover != "cpu" => panic!("SP1_PROVER={prover}: only the CPU prover is allowed"),
        _ => {}
    }
}

fn load_elf(case: &str) -> Result<Elf> {
    let binary = zkvm_kernels::binary_name(case);
    let (_, path) = GUEST_ELFS
        .iter()
        .find(|(name, _)| *name == binary)
        .with_context(|| format!("no SP1 guest for {case}"))?;
    let bytes = std::fs::read(path).with_context(|| format!("reading guest ELF {path}"))?;
    Ok(Elf::from(bytes))
}

/// One raw buffer of little-endian words per kernel argument.
fn stdin(args: &CommonArgs) -> Result<SP1Stdin> {
    let mut stdin = SP1Stdin::new();
    for words in args.kernel_args()? {
        stdin.write_slice(words_as_le_bytes(&words));
    }
    Ok(stdin)
}

fn setup(report: &mut Report, elf: Elf, work: &WorkFiles) -> Result<()> {
    let client = report.time("client_init", CpuProver::new);
    make_keys(report, &client, elf, work)?;
    Ok(())
}

/// Computes the proving and verifying keys and stores them in the work
/// directory (timed as `setup`).
fn make_keys(report: &mut Report, client: &CpuProver, elf: Elf, work: &WorkFiles) -> Result<SP1ProvingKey> {
    let pk = report.time("setup", || client.setup(elf)).context("setup")?;
    save(&work.proving_key, &pk)?;
    save(&work.verifying_key, pk.verifying_key())?;
    report.set("vk_hash", pk.verifying_key().bytes32());
    report.set("vk_path", &work.verifying_key);
    Ok(pk)
}

fn execute(report: &mut Report, elf: Elf, args: &CommonArgs) -> Result<()> {
    let stdin = stdin(args)?;
    let client = report.time("client_init", LightProver::new);
    let (public_values, execution) =
        report.time("execute", || client.execute(elf, stdin).run()).context("execution")?;
    record_execution(report, &execution)?;
    args.report_committed(report, &committed_words(&public_values)?)?;
    Ok(())
}

fn prove(report: &mut Report, elf: Elf, cli: &Cli, work: &WorkFiles) -> Result<()> {
    let args = &cli.common;
    report.set("proof_mode", cli.proof_mode.name());
    let client = report.time("client_init", CpuProver::new);
    // Without a stored key (no `setup` run yet) the keys are made here and
    // stored, so `verify` works after `prove` alone.
    let pk: SP1ProvingKey = match load(&work.proving_key) {
        Ok(pk) => pk,
        Err(_) => make_keys(report, &client, elf.clone(), work)?,
    };

    let (_, execution) =
        report.time("execute", || client.execute(elf, stdin(args)?).run().map_err(anyhow::Error::from))?;
    record_execution(report, &execution)?;

    let stdin = stdin(args)?;
    let proof = report
        .time("prove", || client.prove(&pk, stdin).mode(cli.proof_mode.sdk_mode()).run())
        .map_err(|err| anyhow::anyhow!("proving: {err}"))?;

    let path = work.proof(cli.proof_mode, &args.input);
    proof.save(&path).context("saving proof")?;
    report.set("proof_path", &path);
    report.set("proof_size_bytes", std::fs::metadata(&path)?.len());
    args.report_committed(report, &committed_words(&proof.public_values)?)?;
    Ok(())
}

fn verify(report: &mut Report, cli: &Cli, work: &WorkFiles) -> Result<()> {
    let args = &cli.common;
    report.set("proof_mode", cli.proof_mode.name());
    let path = work.proof(cli.proof_mode, &args.input);
    let mut proof = SP1ProofWithPublicValues::load(&path)
        .with_context(|| format!("loading proof {}", path.display()))?;
    let vk: SP1VerifyingKey = load(&work.verifying_key)?;
    report.set("proof_path", &path);
    report.set("proof_size_bytes", std::fs::metadata(&path)?.len());
    args.report_committed(report, &committed_words(&proof.public_values)?)?;

    if args.tamper {
        let mut bytes = proof.public_values.to_vec();
        tamper_byte(&mut bytes)?;
        proof.public_values = SP1PublicValues::from(&bytes);
    }
    let client = report.time("client_init", LightProver::new);
    let outcome = report.time("verify", || client.verify(&proof, &vk, None));
    report.set("verified", outcome.is_ok());
    match (outcome, args.tamper) {
        (Ok(()), false) | (Err(_), true) => Ok(()),
        (Err(err), false) => bail!("verification failed: {err}"),
        (Ok(()), true) => bail!("a proof with tampered public values verified"),
    }
}

/// Records the execution's size, and fails if the guest did not exit
/// normally: SP1's executor returns a guest panic as an execution with a
/// nonzero exit code (and whatever was committed before it), not as an
/// error.
fn record_execution(report: &mut Report, execution: &sp1_sdk::ExecutionReport) -> Result<()> {
    if execution.exit_code != 0 {
        bail!("guest panicked (exit code {})", execution.exit_code);
    }
    report.set("cycles", execution.total_instruction_count());
    report.set("syscalls", execution.total_syscall_count());
    if let Some(gas) = execution.gas() {
        report.set("gas", gas);
    }
    Ok(())
}

/// The words the guest committed as its public values.
fn committed_words(public_values: &SP1PublicValues) -> Result<Vec<i64>> {
    words_from_le_bytes(public_values.as_slice()).context("the public values are not a whole number of words")
}

fn save<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let bytes = bincode::serialize(value)?;
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(bincode::deserialize(&bytes)?)
}
