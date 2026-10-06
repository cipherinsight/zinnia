//! Deterministic Z3 budgets: every Z3 query on the compile path is bounded by
//! a *resource limit* (Z3's `rlimit`), never by a wall-clock timeout.
//!
//! The compiler acts on Z3's answers (a bound proved or not, an assert kept or
//! dropped, a construction selected or not). Under a wall-clock timeout the
//! answer to a hard query depends on how busy the machine is, so the same
//! source could compile to different circuits on different runs. The `rlimit`
//! counts Z3's internal steps instead: the same sequence of queries uses the
//! same steps on every run and every machine, so it gets the same answers.
//!
//! The steps a query takes also depend on the queries before it in the same
//! Z3 context (Z3 keeps state between queries). Every compilation runs on its
//! own thread (`lib.rs`, `compile_circuit`), hence with a fresh thread-local
//! context, and issues its queries in a deterministic order, so this holds.
//! Z3's other sources of variation are off by default and stay off: its
//! random seeds default to 0 and parallel solving is disabled.
//!
//! Each limit below was calibrated against the timeout it replaces (see
//! [`trace`]). On this machine Z3 takes roughly 1 to 20 million steps per
//! second depending on the query, so each limit is about the old time budget
//! at 10 million steps per second; on the performance set every query takes
//! at most a ninth of its limit, and all are answered as they were under the
//! timeouts on an idle machine.

use std::time::Instant;

use z3::ast::Bool;
use z3::{Optimize, Params, SatResult, Solver, Statistics};

/// Steps for one resolver query during IR generation (`optim::resolver::smt`,
/// `IRGenConfig::smt_query_rlimit`, and the shape-parameter bounds of
/// `optim::prove`); it replaced a 100 ms timeout. On the performance set the
/// costliest such query takes about 1,200 steps.
pub const RESOLVER_RLIMIT: u32 = 1_000_000;

/// Steps for one resolver query of the optimizer
/// (`LayeredResolver::range_then_smt`); it replaced a 500 ms timeout.
pub const OPTIMIZER_RESOLVER_RLIMIT: u32 = 5_000_000;

/// Steps for one entailment query (`optim::prove::prove`); it replaced a
/// 500 ms timeout. On the performance set the costliest takes about 1.1
/// million steps (0.1–0.25 s on an idle machine).
pub const PROVE_RLIMIT: u32 = 10_000_000;

/// Steps for one construction-selection problem (`lower::select`); it
/// replaced a 10 s timeout. On the performance set the costliest takes about
/// 5.8 million steps (0.3–0.5 s).
pub const SELECT_RLIMIT: u32 = 100_000_000;

/// Solver parameters that bound each `check` by `rlimit` Z3 steps.
pub fn params(rlimit: u32) -> Params {
    let mut params = Params::new();
    params.set_u32("rlimit", rlimit);
    params
}

/// A solver whose every `check` is bounded by `rlimit` steps.
pub fn solver(rlimit: u32) -> Solver {
    let solver = Solver::new();
    solver.set_params(&params(rlimit));
    solver
}

/// An optimizer whose every `check` is bounded by `rlimit` steps.
pub fn optimize(rlimit: u32) -> Optimize {
    let opt = Optimize::new();
    opt.set_params(&params(rlimit));
    opt
}

/// `solver.check()`, traced (see [`traced`]).
pub fn check(site: &str, solver: &Solver) -> SatResult {
    traced(site, || solver.get_statistics(), || solver.check())
}

/// `solver.check_assumptions(assumptions)`, traced (see [`traced`]).
pub fn check_assuming(site: &str, solver: &Solver, assumptions: &[Bool]) -> SatResult {
    traced(site, || solver.get_statistics(), || solver.check_assumptions(assumptions))
}

/// `opt.check(&[])`, traced (see [`traced`]).
pub fn check_optimize(site: &str, opt: &Optimize) -> SatResult {
    traced(site, || opt.get_statistics(), || opt.check(&[]))
}

/// Run `check`. With `ZINNIA_Z3_TRACE` set, also print one stderr line: the
/// call site, the answer, the steps the check used and its wall time. This is
/// how the limits above are calibrated; it has no effect on the answers.
fn traced(site: &str, stats: impl Fn() -> Statistics, check: impl FnOnce() -> SatResult) -> SatResult {
    if std::env::var_os("ZINNIA_Z3_TRACE").is_none() {
        return check();
    }
    let before = step_count(&stats());
    let started = Instant::now();
    let result = check();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    let used = step_count(&stats()).saturating_sub(before);
    eprintln!("[zinnia-z3] site={site} result={result:?} rlimit={used} ms={elapsed_ms:.3}");
    result
}

/// The Z3 context's step count so far.
fn step_count(stats: &Statistics) -> u64 {
    match stats.value("rlimit count") {
        Some(z3::StatisticsValue::UInt(n)) => n as u64,
        Some(z3::StatisticsValue::Double(x)) => x as u64,
        None => 0,
    }
}
