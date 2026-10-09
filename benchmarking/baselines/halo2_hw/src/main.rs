//! Command-line driver of the handwritten halo2-base baselines.
//!
//! ```text
//! halo2_hw <mode> --case <case dir or name> [--function <f>] --input <A|B> --workdir <dir>
//!          [--fixture <path>] [--out <json>] [--k <k>] [--lookup-bits <bits>]
//!          [--instance <json>] [--tamper-index <i>] [--repeats <n>]
//!          [--tamper-input <name>:<index>]   (mock: one input out of its domain)
//! halo2_hw list
//! ```
//!
//! Modes: `setup` (size, keys; normally input A), `prove` (normally input B;
//! `--repeats n` proves n times in-process and reports every sample),
//! `verify`, `mock` (MockProver, no keys), `stats` (cell counts and the chosen
//! size). Each run prints one JSON object (and writes it to `--out` if given):
//! `{"ok", "case", "mode", "input", "publics", "proof_path", "timings_s", ...}`.
//! The exit status is non-zero whenever `ok` is false; `verify` reports
//! `ok: false` when the proof is rejected.
//!
//! A DS-1000 case is its folder (`--case .../ds1000/ds1000_case0309`, fixture
//! `fixture.json`) or its name (`0309`, `ds1000_case0309`). An NPBench/Pythran
//! entry is its program folder and entry function (`--case .../pythran/pythran_lu
//! --function factorMatrix0`: case `pythran_lu__factorMatrix0`, fixture
//! `fixture.factorMatrix0.json` if the folder has one, else `fixture.json`) or
//! its case id (`pythran_lu__factorMatrix0`, with `--fixture`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use halo2_hw::fixture::Fixture;
use halo2_hw::scaffold::{self, Timings};
use halo2_hw::{cases, CircuitFn, Error};
use serde_json::{json, Value};

struct Args {
    mode: String,
    flags: HashMap<String, String>,
}

fn parse_args() -> Result<Args, Error> {
    let mut it = std::env::args().skip(1);
    let mode = it.next().ok_or("missing mode")?;
    let mut flags = HashMap::new();
    while let Some(flag) = it.next() {
        let name = flag.strip_prefix("--").ok_or_else(|| Error::new(format!("unexpected argument {flag:?}")))?;
        let value = it.next().ok_or_else(|| Error::new(format!("--{name} needs a value")))?;
        flags.insert(name.to_string(), value);
    }
    Ok(Args { mode, flags })
}

impl Args {
    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }
    fn require(&self, name: &str) -> Result<&str, Error> {
        self.get(name).ok_or_else(|| Error::new(format!("--{name} is required for {}", self.mode)))
    }
    fn number<T: std::str::FromStr>(&self, name: &str) -> Result<Option<T>, Error> {
        self.get(name)
            .map(|v| v.parse().map_err(|_| Error::new(format!("--{name}: not a number: {v}"))))
            .transpose()
    }
}

/// A case to run: its id and, when given as a folder, the folder's fixture.
struct CaseSpec {
    name: String,
    fixture: Option<PathBuf>,
}

/// Resolves `--case` (a folder or a case id) and `--function` (the entry
/// function of a program folder) to the case id and its default fixture.
fn resolve_case(spec: &str, function: Option<&str>) -> CaseSpec {
    let path = Path::new(spec);
    if path.is_dir() {
        let folder = path.file_name().and_then(|n| n.to_str()).unwrap_or(spec);
        return match function {
            None => CaseSpec { name: folder.to_string(), fixture: Some(path.join("fixture.json")) },
            Some(f) => {
                let own = path.join(format!("fixture.{f}.json"));
                let fixture = if own.is_file() { own } else { path.join("fixture.json") };
                CaseSpec { name: format!("{folder}__{f}"), fixture: Some(fixture) }
            }
        };
    }
    let name = if spec.bytes().all(|b| b.is_ascii_digit()) { format!("ds1000_case{spec}") } else { spec.to_string() };
    CaseSpec { name, fixture: None }
}

fn run(args: &Args, report: &mut serde_json::Map<String, Value>) -> Result<bool, Error> {
    if args.mode == "list" {
        report.insert("cases".into(), json!(cases::ALL));
        return Ok(true);
    }
    let CaseSpec { name: case, fixture: folder_fixture } = resolve_case(args.require("case")?, args.get("function"));
    report.insert("case".into(), json!(case));
    let circuit: CircuitFn =
        cases::lookup(&case).ok_or_else(|| Error::new(format!("{case} is not compiled into this build")))?;
    let mut t = Timings::default();

    let load_inputs = |t: &mut Timings| -> Result<_, Error> {
        let input = args.require("input")?;
        let fixture_path = match (args.get("fixture"), &folder_fixture) {
            (Some(p), _) => PathBuf::from(p),
            (None, Some(f)) => f.clone(),
            (None, None) => return Err(Error::new("--fixture is required when --case is not a folder")),
        };
        t.time("load_input_s", || Fixture::load(&fixture_path)?.inputs(input))
    };
    let k = args.number::<u32>("k")?;
    let lookup_bits = args.number::<u32>("lookup-bits")?;

    let (result, ok) = match args.mode.as_str() {
        "setup" => {
            report.insert("input".into(), json!(args.require("input")?));
            let inputs = load_inputs(&mut t)?;
            let workdir = PathBuf::from(args.require("workdir")?);
            (scaffold::setup(&case, circuit, &inputs, &workdir, k, lookup_bits, &mut t)?, true)
        }
        "prove" => {
            report.insert("input".into(), json!(args.require("input")?));
            let inputs = load_inputs(&mut t)?;
            let workdir = PathBuf::from(args.require("workdir")?);
            let repeats = args.number::<usize>("repeats")?.unwrap_or(1);
            (scaffold::prove(&case, circuit, &inputs, &workdir, repeats, &mut t)?, true)
        }
        "verify" => {
            let workdir = PathBuf::from(args.require("workdir")?);
            let instance = args.get("instance").map(PathBuf::from);
            let tamper = args.number::<usize>("tamper-index")?;
            let out = scaffold::verify(&case, &workdir, instance.as_deref(), tamper, &mut t)?;
            let accepted = out["verified"].as_bool() == Some(true);
            (out, accepted)
        }
        "mock" => {
            report.insert("input".into(), json!(args.require("input")?));
            let mut inputs = load_inputs(&mut t)?;
            if let Some(spec) = args.get("tamper-input") {
                let (name, index) = spec
                    .rsplit_once(':')
                    .and_then(|(n, i)| Some((n, i.parse::<usize>().ok()?)))
                    .ok_or_else(|| Error::new(format!("--tamper-input expects name:index, got {spec:?}")))?;
                inputs.tamper(name, index)?;
                report.insert("tampered_input".into(), json!(spec));
            }
            let out = scaffold::mock(circuit, &inputs, k, lookup_bits, &mut t)?;
            let satisfied = out["satisfied"].as_bool() == Some(true);
            (out, satisfied)
        }
        "stats" => {
            report.insert("input".into(), json!(args.require("input")?));
            let inputs = load_inputs(&mut t)?;
            let (k, lb) = t.time("choose_k_s", || scaffold::choose_size(circuit, &inputs, k, lookup_bits))?;
            let cells = scaffold::cell_stats(circuit, &inputs, lb)?;
            (json!({"k": k, "lookup_bits": lb, "cells": cells}), true)
        }
        other => return Err(Error::new(format!("unknown mode {other:?}"))),
    };
    if let Value::Object(fields) = result {
        report.extend(fields);
    }
    report.insert("timings_s".into(), t.into_json());
    Ok(ok)
}

fn main() -> ExitCode {
    let mut report = serde_json::Map::new();
    let mut out_path = None;
    let outcome = parse_args().and_then(|args| {
        report.insert("mode".into(), json!(args.mode));
        out_path = args.get("out").map(PathBuf::from);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut local = serde_json::Map::new();
            let r = run(&args, &mut local);
            (r, local)
        }));
        match caught {
            Ok((r, local)) => {
                report.extend(local);
                r
            }
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                Err(Error::new(format!("panic: {msg}")))
            }
        }
    });
    let ok = outcome.as_ref().map_or(false, |ok| *ok);
    if let Err(e) = outcome {
        report.insert("error".into(), json!(e.0));
    }
    report.insert("ok".into(), json!(ok));
    let text = Value::Object(report).to_string();
    println!("{text}");
    if let Some(path) = out_path {
        if let Err(e) = std::fs::write(&path, &text) {
            eprintln!("cannot write {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    }
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
