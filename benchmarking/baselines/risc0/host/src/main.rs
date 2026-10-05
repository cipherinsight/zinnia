//! RISC Zero host for the DS-1000 baselines (risc0-zkvm 3.0.3, CPU prover in
//! this process).
//!
//! Modes (see `ds1000_zkvm::host` for the program contract):
//! * `setup`: load the guest program and report its image ID. RISC Zero has
//!   no per-program key generation; the verifier needs only the image ID;
//! * `execute`: run the guest in the executor; journal and cycle counts;
//! * `prove`: execute (timed on its own), prove the resulting session, and
//!   compress the receipt as the proof mode asks, each stage timed;
//! * `verify`: verify the stored receipt against the image ID.
//!
//! Proof modes: `composite` (one STARK per segment), `succinct` (segments
//! recursively compressed to one STARK), `groth16` (the succinct receipt
//! wrapped in a Groth16 SNARK; RISC Zero runs that step in Docker).

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use ds1000_zkvm::{
    host::{run_host, tamper_byte, CommonArgs, Mode, Report},
    words_from_le_bytes, Public,
};
use risc0_ds1000_methods::GUESTS;
use risc0_zkvm::{
    get_prover_server, Digest, ExecutorEnv, ExecutorImpl, ProverOpts, Receipt, Session,
    VerifierContext,
};

const ZKVM_VERSION: &str = "3.0.3";

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[arg(long, value_enum, default_value = "composite")]
    proof_mode: ProofMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ProofMode {
    Composite,
    Succinct,
    Groth16,
}

impl ProofMode {
    fn name(self) -> &'static str {
        match self {
            ProofMode::Composite => "composite",
            ProofMode::Succinct => "succinct",
            ProofMode::Groth16 => "groth16",
        }
    }
}

struct Guest {
    program: Vec<u8>,
    image_id: Digest,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    assert!(
        std::env::var_os("RISC0_DEV_MODE").is_none(),
        "RISC0_DEV_MODE is set: dev mode produces fake receipts"
    );
    run_host("risc0", &cli.common, |report| {
        report.set("zkvm_version", ZKVM_VERSION);
        let case = cli.common.case()?;
        let work = cli.common.work_dir(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())?;
        let guest = report.time("load_program", || load_guest(&case))?;
        report.set("image_id", guest.image_id.to_string());
        match cli.common.mode {
            Mode::Setup => Ok(()),
            Mode::Execute => execute(report, &case, &guest, &cli.common).map(|_| ()),
            Mode::Prove => prove(report, &case, &guest, &cli, &work),
            Mode::Verify => verify(report, &case, &guest, &cli, &work),
        }
    })
}

fn load_guest(case: &str) -> Result<Guest> {
    let (_, path, id) = GUESTS
        .iter()
        .find(|(name, _, _)| *name == case)
        .with_context(|| format!("no RISC Zero guest for {case}"))?;
    let program = std::fs::read(path).with_context(|| format!("reading guest program {path}"))?;
    Ok(Guest { program, image_id: Digest::from(*id) })
}

fn proof_path(work: &Path, mode: ProofMode, input: &str) -> PathBuf {
    work.join(format!("receipt-{}-{input}.bin", mode.name()))
}

/// One frame per kernel argument: its word count, then its words.
fn executor_env(args: &CommonArgs) -> Result<ExecutorEnv<'static>> {
    let mut builder = ExecutorEnv::builder();
    for words in args.kernel_args()? {
        let len = u32::try_from(words.len()).context("argument too long")?;
        builder.write_slice(&[len]).write_slice(&words);
    }
    builder.build()
}

/// Runs the guest in the executor and records the cycle counts.
fn execute(report: &mut Report, case: &str, guest: &Guest, args: &CommonArgs) -> Result<Session> {
    let env = executor_env(args)?;
    let session = report.time("execute", || ExecutorImpl::from_elf(env, &guest.program)?.run())?;
    report.set("user_cycles", session.user_cycles);
    report.set("total_cycles", session.total_cycles);
    report.set("paging_cycles", session.paging_cycles);
    report.set("segments", session.segments.len());
    let journal = session.journal.as_ref().context("the guest committed no journal")?;
    report.set_publics(&decode_publics(case, &journal.bytes)?);
    Ok(session)
}

fn prove(report: &mut Report, case: &str, guest: &Guest, cli: &Cli, work: &Path) -> Result<()> {
    let args = &cli.common;
    report.set("proof_mode", cli.proof_mode.name());
    let session = execute(report, case, guest, args)?;

    let prover = get_prover_server(&ProverOpts::composite())?;
    let mut receipt = report
        .time("prove", || prover.prove_session(&VerifierContext::default(), &session))?
        .receipt;
    if cli.proof_mode != ProofMode::Composite {
        receipt = report.time("compress_succinct", || prover.compress(&ProverOpts::succinct(), &receipt))?;
    }
    if cli.proof_mode == ProofMode::Groth16 {
        receipt = report.time("wrap_groth16", || prover.compress(&ProverOpts::groth16(), &receipt))?;
    }

    let path = proof_path(work, cli.proof_mode, &args.input);
    std::fs::write(&path, bincode::serialize(&receipt)?).context("saving receipt")?;
    report.set("proof_path", &path);
    report.set("proof_size_bytes", std::fs::metadata(&path)?.len());
    report.set_publics(&decode_publics(case, &receipt.journal.bytes)?);
    Ok(())
}

fn verify(report: &mut Report, case: &str, guest: &Guest, cli: &Cli, work: &Path) -> Result<()> {
    let args = &cli.common;
    report.set("proof_mode", cli.proof_mode.name());
    let path = proof_path(work, cli.proof_mode, &args.input);
    let bytes = std::fs::read(&path).with_context(|| format!("reading receipt {}", path.display()))?;
    let mut receipt: Receipt = bincode::deserialize(&bytes)?;
    report.set("proof_path", &path);
    report.set("proof_size_bytes", bytes.len());
    report.set_publics(&decode_publics(case, &receipt.journal.bytes)?);

    if args.tamper {
        tamper_byte(&mut receipt.journal.bytes)?;
    }
    let outcome = report.time("verify", || receipt.verify(guest.image_id));
    report.set("verified", outcome.is_ok());
    match (outcome, args.tamper) {
        (Ok(()), false) | (Err(_), true) => Ok(()),
        (Err(err), false) => bail!("verification failed: {err}"),
        (Ok(()), true) => bail!("a receipt with a tampered journal verified"),
    }
}

/// Decodes the journal: the output's words, little-endian.
fn decode_publics(case: &str, journal: &[u8]) -> Result<Vec<Public>> {
    let words = words_from_le_bytes(journal).context("the journal is not a whole number of words")?;
    let decoded = ds1000_zkvm::decode_publics(case, &words)
        .with_context(|| format!("unknown case {case}"))?;
    Ok(decoded?)
}
