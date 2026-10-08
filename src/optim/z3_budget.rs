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

/// Steps for a decision query that only tries to tighten a bound a cheaper
/// tier already gives (`optim::resolver::Effort::Tighten`): `v >= 0` asked of
/// a `v` the intervals put in `[-1, 8]`. Such a check is speculative, so it
/// gets a tenth of [`RESOLVER_RLIMIT`]. The tightenings that succeed on the
/// corpora take at most a few thousand steps; one that fails would otherwise
/// spend the full budget on a formula a loop has made hard.
pub const TIGHTEN_RLIMIT: u32 = 100_000;

/// Steps for one resolver query of the optimizer
/// (`LayeredResolver::range_then_smt`); it replaced a 500 ms timeout.
pub const OPTIMIZER_RESOLVER_RLIMIT: u32 = 5_000_000;

/// Steps for one entailment query (`optim::prove::prove`); it replaced a
/// 500 ms timeout. On the performance set the costliest takes about 1.1
/// million steps (0.1–0.25 s on an idle machine).
pub const PROVE_RLIMIT: u32 = 10_000_000;

/// Steps for all value-reasoning queries of one compilation together (every
/// query except the construction selection). Each query is bounded by its own
/// limit above, but a program can issue thousands of them: Pythran `ccopy`
/// issues ~1,000 bound queries during IR generation, two thirds of which use
/// their whole million steps and answer `Unknown`, so the compilation spent
/// minutes learning nothing. Once a compilation has used this many steps, every
/// further reasoning query answers `Unknown` without calling Z3, exactly as
/// under [`smt_ablated`]: the callers' fallbacks (intervals, a check emitted in
/// the circuit) take over, so the circuit stays sound and the compilation's
/// cost is bounded. The count is per compilation thread and the queries come in
/// a deterministic order, so whether and where the budget runs out is the same
/// on every run. Override with `ZINNIA_SMT_TOTAL_RLIMIT` (measurement).
pub const TOTAL_REASONING_RLIMIT: u64 = 200_000_000;

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

/// Measurement only: `ZINNIA_ABLATE_SMT=1` turns off SMT reasoning
/// everywhere on the compile path. Every reasoning query below then answers
/// `Unknown` without calling Z3, which each caller already treats as "no
/// information" (the soundness rule of `optim::prove`): the resolvers fall
/// back to their interval layer, `prove` to its interval fast paths, and a
/// precondition nothing else proves is checked in the circuit. The
/// construction selection ([`check_selection`]) is an optimization problem,
/// not reasoning about values, and is not affected. Unset (the default), it
/// has no effect. Read once per process.
pub fn smt_ablated() -> bool {
    static ABLATED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ABLATED.get_or_init(|| std::env::var("ZINNIA_ABLATE_SMT").is_ok_and(|v| v.trim() == "1"))
}

fn total_reasoning_rlimit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("ZINNIA_SMT_TOTAL_RLIMIT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(TOTAL_REASONING_RLIMIT)
    })
}

thread_local! {
    /// Steps the reasoning queries of this thread's compilation have used.
    static REASONING_STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Run a value-reasoning query: `Unknown` without calling Z3 under
/// [`smt_ablated`] or once this compilation's [`TOTAL_REASONING_RLIMIT`] is
/// spent; otherwise the query (traced), with its steps charged to the budget.
fn reasoning(site: &str, stats: impl Fn() -> Statistics, check: impl FnOnce() -> SatResult) -> SatResult {
    if smt_ablated() {
        return SatResult::Unknown;
    }
    let spent = REASONING_STEPS.with(|s| s.get());
    if spent >= total_reasoning_rlimit() {
        if std::env::var_os("ZINNIA_Z3_TRACE").is_some() {
            eprintln!("[zinnia-z3] site={site} result=Unknown rlimit=0 ms=0 (compilation budget spent)");
        }
        return SatResult::Unknown;
    }
    let before = step_count(&stats());
    let result = traced(site, &stats, check);
    let used = step_count(&stats()).saturating_sub(before);
    REASONING_STEPS.with(|s| s.set(spent.saturating_add(used)));
    result
}

/// `solver.check()`, traced (see [`traced`]); `Unknown` under [`smt_ablated`]
/// or once the compilation's reasoning budget is spent.
pub fn check(site: &str, solver: &Solver) -> SatResult {
    reasoning(site, || solver.get_statistics(), || solver.check())
}

/// `solver.check_assumptions(assumptions)`, traced (see [`traced`]);
/// `Unknown` under [`smt_ablated`] or once the compilation's reasoning budget
/// is spent.
pub fn check_assuming(site: &str, solver: &Solver, assumptions: &[Bool]) -> SatResult {
    reasoning(site, || solver.get_statistics(), || solver.check_assumptions(assumptions))
}

/// `opt.check(&[])` for a bound query (the extreme of a value), traced (see
/// [`traced`]); `Unknown` under [`smt_ablated`] or once the compilation's
/// reasoning budget is spent.
pub fn check_optimize(site: &str, opt: &Optimize) -> SatResult {
    reasoning(site, || opt.get_statistics(), || opt.check(&[]))
}

/// `opt.check(&[])` for the construction selection (`lower::select`), traced
/// (see [`traced`]). Not reasoning about values, so [`smt_ablated`] leaves it on.
pub fn check_selection(site: &str, opt: &Optimize) -> SatResult {
    traced(site, &|| opt.get_statistics(), || opt.check(&[]))
}

/// Run `check`. With `ZINNIA_Z3_TRACE` set, also print one stderr line: the
/// call site, the answer, the steps the check used and its wall time. This is
/// how the limits above are calibrated; it has no effect on the answers.
fn traced(site: &str, stats: &impl Fn() -> Statistics, check: impl FnOnce() -> SatResult) -> SatResult {
    if std::env::var_os("ZINNIA_Z3_TRACE").is_none() {
        return check();
    }
    let before = step_count(&stats());
    let started = Instant::now();
    let result = check();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    let used = step_count(&stats()).saturating_sub(before);
    let caller = if std::env::var_os("ZINNIA_Z3_TRACE_CALLER").is_some() { caller_frames() } else { String::new() };
    eprintln!("[zinnia-z3] site={site} result={result:?} rlimit={used} ms={elapsed_ms:.3}{caller}");
    result
}

/// Measurement only (`ZINNIA_Z3_TRACE_CALLER`): the first compiler frames
/// outside the reasoning modules that led to a query, as ` caller=a<b<c`.
fn caller_frames() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let frames: Vec<String> = bt
        .lines()
        .filter_map(|l| l.trim().split_once(": ").map(|(_, f)| f.to_string()))
        .filter(|f| f.starts_with("_zinnia_core::"))
        .filter(|f| !["optim::resolver", "optim::z3_budget", "optim::prove", "builder::IRBuilder>::ask_bounds", "optim::range"].iter().any(|m| f.contains(m)))
        .map(|f| f.trim_start_matches("_zinnia_core::").split("::h").next().unwrap_or("").to_string())
        .take(3)
        .collect();
    format!(" caller={}", frames.join("<"))
}

/// The Z3 context's step count so far.
fn step_count(stats: &Statistics) -> u64 {
    match stats.value("rlimit count") {
        Some(z3::StatisticsValue::UInt(n)) => n as u64,
        Some(z3::StatisticsValue::Double(x)) => x as u64,
        None => 0,
    }
}
