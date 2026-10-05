//! Host-side pieces shared by the SP1 and RISC Zero hosts and the native
//! runner: the command line of the program contract, the JSON report, and
//! running a case natively.
//!
//! Program contract (see `eval-redo/BRIEF.md`): a host is driven with the case
//! folder, the fixture, the input name (`A` or `B`) and a mode, and prints one
//! JSON object (also written to `--out` if given) with at least `ok`,
//! `publics`, `proof_path` and `timings_s`. It exits non-zero on any failure.

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::{fixture::Fixture, Input, Public};

/// The arguments every host takes.
#[derive(Args, Debug)]
pub struct CommonArgs {
    /// The case folder, `benchmarking/datasets/ds1000/<case>`.
    #[arg(long)]
    pub case_dir: PathBuf,
    /// The fixture file (default: `<case-dir>/fixture.json`).
    #[arg(long)]
    pub fixture: Option<PathBuf>,
    /// The fixture input: `A` (setup) or `B` (timed proofs, output check).
    #[arg(long, default_value = "B")]
    pub input: String,
    #[arg(long, value_enum)]
    pub mode: Mode,
    /// Where keys and proofs are kept between modes (default:
    /// `<system>/work/<case>` next to the host's crate).
    #[arg(long)]
    pub work_dir: Option<PathBuf>,
    /// Also write the JSON report to this file.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// In `verify` mode: change the last byte of the public values before
    /// verifying. The run succeeds only if verification rejects the proof.
    #[arg(long)]
    pub tamper: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Key generation (independent of the input for zkVMs).
    Setup,
    /// Run the guest in the zkVM's executor only: public values and cycles.
    Execute,
    /// Execute and prove; writes the proof to the work directory.
    Prove,
    /// Verify the proof in the work directory.
    Verify,
}

impl CommonArgs {
    /// The case name, i.e. the case folder's name.
    pub fn case(&self) -> Result<String> {
        let name = self.case_dir.file_name().and_then(|n| n.to_str());
        Ok(name.context("case folder has no name")?.to_string())
    }

    pub fn fixture_path(&self) -> PathBuf {
        self.fixture.clone().unwrap_or_else(|| self.case_dir.join("fixture.json"))
    }

    /// The encoded kernel arguments of the chosen fixture input, without the
    /// parameters the case never reads.
    pub fn kernel_args(&self) -> Result<Vec<Vec<i64>>> {
        let unused = crate::unused_params(&self.case()?);
        Fixture::load(&self.fixture_path())?.args(&self.input, unused)
    }

    /// The work directory, created if missing.
    pub fn work_dir(&self, default_root: &Path) -> Result<PathBuf> {
        let dir = match &self.work_dir {
            Some(dir) => dir.clone(),
            None => default_root.join("work").join(self.case()?),
        };
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(dir)
    }
}

/// The JSON object a host prints.
pub struct Report {
    fields: Map<String, Value>,
    timings: Map<String, Value>,
}

impl Report {
    pub fn new(system: &str, args: &CommonArgs) -> Report {
        let mut fields = Map::new();
        fields.insert("system".into(), json!(system));
        fields.insert("case".into(), json!(args.case().unwrap_or_default()));
        fields.insert("input".into(), json!(args.input));
        fields.insert("mode".into(), json!(format!("{:?}", args.mode).to_lowercase()));
        Report { fields, timings: Map::new() }
    }

    pub fn set(&mut self, key: &str, value: impl Serialize) {
        self.fields.insert(key.into(), json!(value));
    }

    /// Records the output values and, if the output has a data-dependent
    /// shape, that shape (`output_shape`).
    pub fn set_publics(&mut self, publics: &[Public]) {
        self.fields.insert("publics".into(), publics_json(publics));
        let dims: Vec<i64> = publics
            .iter()
            .filter_map(|p| match *p {
                Public::Dim(d) => Some(d),
                _ => None,
            })
            .collect();
        if !dims.is_empty() {
            self.fields.insert("output_shape".into(), json!(dims));
        }
    }

    /// Runs `step` and records its wall-clock time under `timings_s.<name>`.
    pub fn time<T>(&mut self, name: &str, step: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let value = step();
        self.timings.insert(name.into(), json!(start.elapsed().as_secs_f64()));
        value
    }

    fn finish(mut self, outcome: Result<()>) -> Value {
        if let Err(err) = &outcome {
            self.fields.insert("error".into(), json!(format!("{err:#}")));
        }
        self.fields.insert("ok".into(), json!(outcome.is_ok()));
        self.fields.entry("publics").or_insert(json!([]));
        self.fields.entry("proof_path").or_insert(Value::Null);
        self.fields.insert("timings_s".into(), Value::Object(self.timings));
        if let Some(peak) = peak_rss_mib() {
            self.fields.insert("peak_rss_mib".into(), json!(peak));
        }
        Value::Object(self.fields)
    }
}

/// This process's peak resident memory (`VmHWM`), in MiB. Work done in other
/// processes (e.g. RISC Zero's Docker-based Groth16 step) is not included.
fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some((kib / 1024.0).round())
}

/// Runs a host: builds the report, runs `body`, prints the JSON and turns
/// any error into a non-zero exit code.
pub fn run_host(system: &str, args: &CommonArgs, body: impl FnOnce(&mut Report) -> Result<()>) -> ExitCode {
    let mut report = Report::new(system, args);
    let outcome = body(&mut report);
    let failed = outcome.is_err();
    let json = report.finish(outcome);
    println!("{json}");
    if let Some(out) = &args.out {
        if let Err(err) = std::fs::write(out, format!("{json:#}\n")) {
            eprintln!("cannot write {}: {err}", out.display());
            return ExitCode::FAILURE;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// What `--tamper` does: flip the lowest bit of the last byte of the public
/// values, which belongs to the last output value.
pub fn tamper_byte(public_values: &mut [u8]) -> Result<()> {
    let last = public_values.last_mut().context("no public values to tamper with")?;
    *last ^= 1;
    Ok(())
}

/// The private input of a native run: the argument buffers in order.
pub struct NativeInput {
    buffers: VecDeque<Vec<i64>>,
}

impl NativeInput {
    pub fn new(args: Vec<Vec<i64>>) -> NativeInput {
        NativeInput { buffers: args.into() }
    }

    /// Whether the guest read every argument.
    pub fn is_exhausted(&self) -> bool {
        self.buffers.is_empty()
    }
}

impl Input for NativeInput {
    fn read_words(&mut self) -> Vec<i64> {
        self.buffers.pop_front().expect("the guest read past the last argument")
    }
}

/// Runs `case` natively on `args` and decodes its committed words the way
/// the hosts do; `None` if the case is unknown.
pub fn run_native(case: &str, args: Vec<Vec<i64>>) -> Option<Result<Vec<Public>>> {
    let mut input = NativeInput::new(args);
    let words = crate::run(case, &mut input)?;
    if !input.is_exhausted() {
        return Some(Err(anyhow::anyhow!("the guest did not read every argument")));
    }
    let publics = crate::decode_publics(case, &words)?;
    Some(publics.map_err(|err| anyhow::anyhow!("decoding the committed words: {err}")))
}

/// Output values as JSON (floats decoded from Q32, integers, booleans),
/// without the extents of a data-dependent shape.
pub fn publics_json(publics: &[Public]) -> Value {
    Value::Array(
        publics
            .iter()
            .filter_map(|p| match *p {
                Public::Int(v) => Some(json!(v)),
                Public::Float(v) => Some(json!(v.to_f64())),
                Public::Bool(v) => Some(json!(v)),
                Public::Dim(_) => None,
            })
            .collect(),
    )
}
