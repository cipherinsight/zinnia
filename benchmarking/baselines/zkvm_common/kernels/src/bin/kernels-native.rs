//! Runs a case's Q32 program natively (no zkVM) on one fixture input and
//! prints its public values as JSON. Used by the output check; only the
//! `execute` mode is meaningful.

use std::process::ExitCode;

use anyhow::ensure;
use clap::Parser;
use zkvm_kernels::host::{run_host, run_native, CommonArgs, Mode};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let args = &cli.common;
    run_host("native", args, |report| {
        ensure!(args.mode == Mode::Execute, "the native runner only supports --mode execute");
        let case = args.case()?;
        let kernel_args = args.kernel_args()?;
        let words = report.time("execute", || run_native(case, kernel_args))?;
        args.report_committed(report, &words)
    })
}
