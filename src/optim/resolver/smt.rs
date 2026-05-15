//! P1 — discharges constancy / max / min queries via Z3 when the cheap
//! `static_val` fast path can't.
//!
//! ## Design overview
//!
//! * Per-ptr cache (`HashMap<StmtId, ResolvedValue>`). Both `Int(n)` /
//!   `Bool(b)` resolutions and `Unknown` outcomes are cached so a repeat
//!   query on the same wire is free. `on_ir_mutated` clears the cache (P1
//!   conservative; P5 may refine).
//!
//! * Lazy formula construction. Every `resolve_*` query traverses the
//!   reverse-reachability subgraph rooted at `val.stmt_id()` and encodes only
//!   those statements as Z3 constraints — never the whole graph. Reference:
//!   the old Python `_build_smt_constraints_for(ptr)`.
//!
//! * Time budget. Each Z3 query sets the solver's `timeout` parameter
//!   (default 500 ms; configurable via `SmtResolver::with_timeout`). On Z3
//!   `unknown` we cache `Unknown` and return `None`.
//!
//! * Disable flag. `with_disabled(true)` makes every query return `None`
//!   after the static_val fast path. Lets users diagnose whether SMT is the
//!   compile-time bottleneck.
//!
//! * Cached resolutions encoded as literals. When the reverse-reachability
//!   walk encounters a statement whose ptr is already in the cache, it
//!   emits the cached constant as a Z3 literal instead of recursing — keeps
//!   formulas small (paper "encode cached resolutions as literals").
//!
//! ## Z3 lifetime / Send+Sync
//!
//! `z3` 0.20 uses an implicit thread-local `Context`. The solver, all asts,
//! etc. are bound to that context — but the context itself is a `Rc`, and
//! `Rc` is `!Send + !Sync`. Our `Resolver` trait requires `Send + Sync`
//! (because `IRGraph` is held by a `#[pyclass]`). The way out:
//!
//! * `SmtResolver` does NOT store any Z3 state across calls. Each query
//!   constructs a fresh `Solver` (cheap), encodes the formula, runs it,
//!   discards the solver. No `Rc<Context>` ever crosses a method boundary.
//!
//! * The "single Z3 context per compilation" requirement from the spec is
//!   satisfied implicitly: z3 0.20's thread-local context is created lazily
//!   on first use per thread. Compilation is single-threaded, so all
//!   queries from the same compilation share one thread-local context.
//!   Per-query setup cost is minimal (Z3 reuses the context's symbol pool
//!   etc.).
//!
//! * Only the cache and the per-resolver knobs (timeout, disabled, counter)
//!   live in the resolver across calls. All `Send + Sync`.
//!
//! ## IRBuilder/IRGraph handoff (option (b))
//!
//! Each phase of compilation has its own `SmtResolver` (or none). Per the
//! spec's "(b) Each phase has its own SmtResolver with its own cache.
//! Caches don't migrate." This is acceptable for P1; P3+ may share state if
//! profiling shows the cost matters.
//!
//! The IR-graph is passed in via the `_with_stmts` trait methods (see the
//! trait definition above). `IRBuilder::split_resolver_and_stmts(&mut)`
//! and `IRGraph::split_resolver_and_stmts(&mut)` are the chokepoints.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::ir::IRStatement;
use crate::optim::telemetry::SmtTelemetry;
use crate::types::{StmtId, Value};

use super::Resolver;

/// One cached resolution outcome for a wire's ptr.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ResolvedValue {
    Int(i64),
    Bool(bool),
    /// Proved unresolvable (timeout, non-unique model, or off-the-int-path
    /// IR op). Cached so we don't re-query Z3 on the same ptr.
    Unknown,
}

/// Inner resolver state, guarded by a Mutex so the outer `SmtResolver` can
/// be `Send + Sync`. The Mutex is only contended in pathological multi-
/// thread reentry; in single-threaded compilation it's a near-free CAS.
#[derive(Debug, Default)]
struct SmtResolverInner {
    cache: HashMap<StmtId, ResolvedValue>,
}

/// Z3-backed [`Resolver`].
#[derive(Debug)]
pub struct SmtResolver {
    inner: Mutex<SmtResolverInner>,
    timeout_ms: u64,
    disabled: bool,
    /// P5 commit 3: cap on the number of IR statements the reverse-
    /// reachability walk visits per query. `usize::MAX` = unbounded
    /// (legacy P1 behaviour). When the cap is hit the walk aborts and
    /// the resolver returns `None`, counted as
    /// `queries_skipped_oversized` in telemetry.
    max_formula_size: usize,
    /// P5 telemetry. Shared across the layered resolver so range and SMT
    /// counters land in one summary. Defaults to a fresh, isolated
    /// instance (so an `SmtResolver` constructed in a test sees its own
    /// counters).
    telemetry: Arc<SmtTelemetry>,
}

impl Default for SmtResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SmtResolver {
    /// Build a resolver with default knobs (P5: 100 ms timeout, no
    /// formula-size cap, enabled). Note: the IRGenConfig default for
    /// `smt_query_timeout_ms` matches; tests that construct an
    /// `SmtResolver` directly inherit this. If you need the legacy
    /// 500 ms budget for a test, call `.with_timeout(500)`.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(SmtResolverInner::default()),
            timeout_ms: 100,
            disabled: false,
            max_formula_size: usize::MAX,
            telemetry: SmtTelemetry::new(),
        }
    }

    /// Override the per-query Z3 timeout.
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// P5 commit 3: cap the per-query reverse-reachability walk at
    /// `n` IR statements. Larger formulas abort early.
    pub fn with_max_formula_size(mut self, n: usize) -> Self {
        self.max_formula_size = n;
        self
    }

    /// Force every query to return `None` (after the static-val fast
    /// path). Lets users diagnose whether SMT is the compile-time
    /// bottleneck.
    pub fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Swap in a shared telemetry. Used by `LayeredResolver::with_telemetry`
    /// so range and SMT counters end up in the same summary.
    pub fn with_telemetry(mut self, telemetry: Arc<SmtTelemetry>) -> Self {
        self.telemetry = telemetry;
        self
    }

    /// Borrow the shared telemetry handle (e.g. for end-of-compilation
    /// snapshotting).
    pub fn telemetry(&self) -> Arc<SmtTelemetry> {
        Arc::clone(&self.telemetry)
    }

    /// Resolve `val` against the supplied IR statements. The
    /// dispatch is shared by `resolve_int_with_stmts` /
    /// `resolve_bool_with_stmts`, and parameterized over the expected
    /// outcome type.
    fn resolve_inner(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
        want_bool: bool,
    ) -> Option<ResolvedValue> {
        // Note: `queries_total` and `queries_static_val_hit` are counted at
        // the LayeredResolver level so they don't double-count when the
        // range layer also touches them. SmtResolver only records the
        // SMT-specific counters below.

        // Fast path 1: static-val.
        if !want_bool {
            if let Some(n) = val.int_val() {
                return Some(ResolvedValue::Int(n));
            }
        } else if let Some(b) = val.bool_val() {
            return Some(ResolvedValue::Bool(b));
        }

        // Fast path 2: ptr-cache hit.
        let ptr = val.stmt_id()?;
        {
            let inner = self.inner.lock().unwrap();
            if let Some(cached) = inner.cache.get(&ptr) {
                self.telemetry.queries_cache_hit.fetch_add(1, Ordering::Relaxed);
                return Some(*cached);
            }
        }

        // Disable flag: short-circuit to Unknown after the static-val
        // fast path. Cache + return.
        if self.disabled {
            self.telemetry.queries_skipped_disabled.fetch_add(1, Ordering::Relaxed);
            self.cache_outcome(ptr, ResolvedValue::Unknown);
            return Some(ResolvedValue::Unknown);
        }

        // Build the formula via reverse-reachability. Time it for the
        // duration histogram + total-time counter.
        let t0 = Instant::now();
        let max_size = if self.max_formula_size == usize::MAX {
            None
        } else {
            Some(self.max_formula_size)
        };
        let qout = smt_query_with_budget(
            ptr,
            stmts,
            &self.inner.lock().unwrap().cache,
            self.timeout_ms,
            want_bool,
            max_size,
        );
        let dur = t0.elapsed();
        self.telemetry.record_smt_duration(dur);
        self.telemetry.note_formula_size(qout.formula_size);

        if qout.oversized {
            self.telemetry
                .queries_skipped_oversized
                .fetch_add(1, Ordering::Relaxed);
        } else {
            match &qout.resolved {
                Some(ResolvedValue::Int(_)) | Some(ResolvedValue::Bool(_)) => {
                    self.telemetry.queries_smt_resolved.fetch_add(1, Ordering::Relaxed);
                }
                Some(ResolvedValue::Unknown) | None => {
                    self.telemetry.queries_smt_unknown.fetch_add(1, Ordering::Relaxed);
                    // Heuristic: if the wall time is ≥ 90 % of the configured
                    // budget, treat it as a timeout for the dedicated counter.
                    // Z3's `unknown` doesn't distinguish timeout vs other
                    // give-ups via the public API, but the duration is a
                    // strong signal in practice.
                    let budget_ns = (self.timeout_ms as u128) * 1_000_000;
                    if budget_ns > 0 && dur.as_nanos() * 10 >= budget_ns * 9 {
                        self.telemetry.queries_smt_timeout.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        let resolved = qout.resolved.unwrap_or(ResolvedValue::Unknown);
        self.cache_outcome(ptr, resolved);
        Some(resolved)
    }

    fn cache_outcome(&self, ptr: StmtId, value: ResolvedValue) {
        let mut inner = self.inner.lock().unwrap();
        inner.cache.insert(ptr, value);
    }

    /// Test-only: count of cached entries. Used by the cache-hit test.
    #[cfg(test)]
    pub fn cache_size(&self) -> usize {
        self.inner.lock().unwrap().cache.len()
    }
}

impl Resolver for SmtResolver {
    /// Without `&[IRStatement]` we can't walk the IR — fall back to
    /// `static_val`. P3+ call sites should route through
    /// `resolve_int_with_stmts` / the IRBuilder split-borrow helper.
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }

    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        val.bool_val()
    }

    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }

    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }

    fn resolve_int_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        match self.resolve_inner(val, stmts, /*want_bool=*/ false)? {
            ResolvedValue::Int(n) => Some(n),
            ResolvedValue::Bool(b) => Some(if b { 1 } else { 0 }),
            ResolvedValue::Unknown => None,
        }
    }

    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<bool> {
        match self.resolve_inner(val, stmts, /*want_bool=*/ true)? {
            ResolvedValue::Bool(b) => Some(b),
            ResolvedValue::Int(n) => Some(n != 0),
            ResolvedValue::Unknown => None,
        }
    }

    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        // Static-val first.
        if let Some(n) = val.int_val() {
            return Some(n);
        }

        // Cache hit short-circuits to the resolved point (max of a
        // unique value is the value).
        let ptr = val.stmt_id()?;
        {
            let inner = self.inner.lock().unwrap();
            if let Some(ResolvedValue::Int(n)) = inner.cache.get(&ptr) {
                self.telemetry.queries_cache_hit.fetch_add(1, Ordering::Relaxed);
                return Some(*n);
            }
        }

        if self.disabled {
            self.telemetry.queries_skipped_disabled.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let max_size = if self.max_formula_size == usize::MAX {
            None
        } else {
            Some(self.max_formula_size)
        };
        let t0 = Instant::now();
        let (resolved, formula_size, oversized) = smt_extreme(
            ptr,
            stmts,
            &self.inner.lock().unwrap().cache,
            self.timeout_ms,
            /*max=*/ true,
            max_size,
        );
        let dur = t0.elapsed();
        self.telemetry.record_smt_duration(dur);
        self.telemetry.note_formula_size(formula_size);
        if oversized {
            self.telemetry.queries_skipped_oversized.fetch_add(1, Ordering::Relaxed);
        } else if resolved.is_some() {
            self.telemetry.queries_smt_resolved.fetch_add(1, Ordering::Relaxed);
        } else {
            self.telemetry.queries_smt_unknown.fetch_add(1, Ordering::Relaxed);
        }
        resolved
    }

    fn resolve_min_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        if let Some(n) = val.int_val() {
            return Some(n);
        }
        let ptr = val.stmt_id()?;
        {
            let inner = self.inner.lock().unwrap();
            if let Some(ResolvedValue::Int(n)) = inner.cache.get(&ptr) {
                self.telemetry.queries_cache_hit.fetch_add(1, Ordering::Relaxed);
                return Some(*n);
            }
        }
        if self.disabled {
            self.telemetry.queries_skipped_disabled.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let max_size = if self.max_formula_size == usize::MAX {
            None
        } else {
            Some(self.max_formula_size)
        };
        let t0 = Instant::now();
        let (resolved, formula_size, oversized) = smt_extreme(
            ptr,
            stmts,
            &self.inner.lock().unwrap().cache,
            self.timeout_ms,
            /*max=*/ false,
            max_size,
        );
        let dur = t0.elapsed();
        self.telemetry.record_smt_duration(dur);
        self.telemetry.note_formula_size(formula_size);
        if oversized {
            self.telemetry.queries_skipped_oversized.fetch_add(1, Ordering::Relaxed);
        } else if resolved.is_some() {
            self.telemetry.queries_smt_resolved.fetch_add(1, Ordering::Relaxed);
        } else {
            self.telemetry.queries_smt_unknown.fetch_add(1, Ordering::Relaxed);
        }
        resolved
    }

    fn on_ir_mutated(&mut self, _affected: &[StmtId]) {
        // P1 conservative: blow the entire cache. P5 may refine to precise
        // ids when profiling shows the cache-rebuild cost matters.
        self.inner.lock().unwrap().cache.clear();
    }

    fn telemetry_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        Some(Arc::clone(&self.telemetry))
    }
}

// ---------------------------------------------------------------------------
// Reverse-reachability walk + Z3 query — free functions to avoid borrow
// nesting issues with `&self` and the cache lock.
// ---------------------------------------------------------------------------

/// Outcome categorisation for `smt_query_with_budget`. Wraps the resolved
/// value (or the reason we didn't resolve) plus the formula size (statement
/// count walked).
struct SmtQueryOut {
    resolved: Option<ResolvedValue>,
    /// Number of distinct IR statements visited by the walker. Used to
    /// populate `largest_formula_size` in the telemetry; also reflected
    /// in the early-abort path when a formula-size budget is set.
    formula_size: usize,
    /// True if the walker aborted because the formula exceeded
    /// `max_formula_size`. Lets the caller bump `queries_skipped_oversized`.
    oversized: bool,
}

/// Build a Z3 formula constraining the wire at `root` and check whether it
/// has a unique value, with an optional cap on formula size. Returns
/// `Some(ResolvedValue::Int|Bool)` if Z3 proves uniqueness within the time
/// budget; `None` on timeout / non-unique. When the reverse-reachability
/// walk exceeds `max_formula_size` IR statements, the walker aborts and the
/// resolver returns `None` — paying only the walk cost, not Z3 time, on the
/// heaviest queries.
///
/// Mirrors the "find a model, then add `expr != that_value` and re-check"
/// pattern from the old Python `SMTUtils.resolve_expr`.
fn smt_query_with_budget(
    root: StmtId,
    stmts: &[IRStatement],
    cache: &HashMap<StmtId, ResolvedValue>,
    timeout_ms: u64,
    want_bool: bool,
    max_formula_size: Option<usize>,
) -> SmtQueryOut {
    let mut walker = Walker::new(stmts, cache);
    walker.max_size = max_formula_size;
    let root_term = match walker.encode(root) {
        Some(t) => t,
        None => {
            return SmtQueryOut {
                resolved: None,
                formula_size: walker.visited,
                oversized: walker.aborted_oversized,
            };
        }
    };
    let formula_size = walker.visited;
    if walker.aborted_oversized {
        return SmtQueryOut {
            resolved: None,
            formula_size,
            oversized: true,
        };
    }

    let solver = z3::Solver::new();
    {
        let mut params = z3::Params::new();
        // Z3's standard "timeout" parameter, in milliseconds.
        params.set_u32("timeout", timeout_ms.min(u32::MAX as u64) as u32);
        solver.set_params(&params);
    }
    for c in walker.constraints.drain(..) {
        solver.assert(&c);
    }
    // Structural-predicate integration (foundation contracts card).
    // Surface all `IR::StructuralPredicate` atoms — they are global facts
    // that constrain the program regardless of reachability from `root`.
    // The discharger assembles their Z3 clauses + meta-facts; the input-
    // name bridge reuses the walker's encodings for the actual SSA Z3
    // terms, so e.g. `nnz(x) == k` ties to the real `k` symbol the
    // walker produced for the `ReadInteger` statement.
    {
        use crate::optim::predicates::{
            build_input_array_lengths, build_input_name_index, Discharger,
        };
        use z3::ast::Int;

        let input_idx = build_input_name_index(stmts);
        let input_lengths = build_input_array_lengths(stmts);

        // Pre-resolve each scalar input's Z3 Int via the walker so we can
        // close over a plain HashMap (Discharger's closure can't hold
        // the walker mutably — borrow-check).
        let mut name_to_int: HashMap<String, Int> = HashMap::new();
        for (name, producer_id) in &input_idx {
            if let Some(term) = walker.encode(*producer_id) {
                name_to_int.insert(name.clone(), term.as_int());
            }
        }

        let discharger = Discharger::new();
        let pred_c = discharger.collect_predicate_constraints(stmts, &input_lengths, |name| {
            if let Some(t) = name_to_int.get(name) {
                t.clone()
            } else if let Ok(n) = name.parse::<i64>() {
                Int::from_i64(n)
            } else {
                // Unknown name (not a scalar input, not a literal).
                // Mint a fresh symbolic; the predicate's facts still
                // hold but they're not tied to a specific SSA.
                Int::fresh_const(&format!("sp_unknown_{name}_"))
            }
        });
        for c in &pred_c.clauses {
            solver.assert(c);
        }
        for f in &pred_c.meta_facts {
            solver.assert(f);
        }

        // Scalar-precondition discharge: deserialize each
        // ContractTerm-shaped atom, lower it to a Z3 Bool via the
        // existing `formula::lower_bool` infrastructure (using the
        // already-built name_to_int as the substitution), and assert
        // on the solver.
        use crate::optim::predicates::{find_scalar_preconditions, formula};
        for term_json in find_scalar_preconditions(stmts) {
            let term: formula::ContractTerm =
                match serde_json::from_str(term_json) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
            let mut subst = formula::Substitution::new();
            for (name, int_term) in &name_to_int {
                subst = subst.with_input(name.clone(), int_term.clone());
            }
            if let Ok(out) = formula::lower_bool(&term, &subst) {
                solver.assert(&out.term);
                for (_, facts) in &out.meta_fact_sets {
                    for f in facts {
                        solver.assert(f);
                    }
                }
            }
        }
    }

    let resolved = match solver.check() {
        z3::SatResult::Sat => {
            // Got a model; ask if `root != model_value` is satisfiable. If
            // unsat, the value is unique → return it.
            solver.get_model().and_then(|model| {
                match (&root_term, want_bool) {
                    (crate::optim::smt_encoding::Z3Term::Int(int), false) => {
                        let v = model.eval(int, true)?;
                        let n = v.as_i64()?;
                        solver.assert(&int.eq(&z3::ast::Int::from_i64(n)).not());
                        if solver.check() == z3::SatResult::Unsat {
                            Some(ResolvedValue::Int(n))
                        } else {
                            None
                        }
                    }
                    (crate::optim::smt_encoding::Z3Term::Int(int), true) => {
                        // Wanted bool, but root is Int. Use `int != 0` as the
                        // bool projection.
                        let zero = z3::ast::Int::from_i64(0);
                        let bool_proj = int.eq(&zero).not();
                        let v = model.eval(&bool_proj, true)?;
                        let b = v.as_bool()?;
                        solver.assert(&bool_proj.eq(&z3::ast::Bool::from_bool(b)).not());
                        if solver.check() == z3::SatResult::Unsat {
                            Some(ResolvedValue::Bool(b))
                        } else {
                            None
                        }
                    }
                    (crate::optim::smt_encoding::Z3Term::Bool(b_ast), _) => {
                        let v = model.eval(b_ast, true)?;
                        let b = v.as_bool()?;
                        solver.assert(&b_ast.eq(&z3::ast::Bool::from_bool(b)).not());
                        if solver.check() == z3::SatResult::Unsat {
                            if want_bool {
                                Some(ResolvedValue::Bool(b))
                            } else {
                                Some(ResolvedValue::Int(if b { 1 } else { 0 }))
                            }
                        } else {
                            None
                        }
                    }
                    // Real wires aren't resolved by the integer/bool
                    // SMT path — Real arithmetic can be unbounded /
                    // irrational, so we conservatively return None.
                    (crate::optim::smt_encoding::Z3Term::Real(_), _) => None,
                }
            })
        }
        z3::SatResult::Unsat | z3::SatResult::Unknown => None,
    };
    SmtQueryOut {
        resolved,
        formula_size,
        oversized: false,
    }
}

/// Discharge an `Optimize` query: maximise (or minimise) the wire at
/// `root` over the constraints from its reverse-reachable subgraph.
///
/// Returns `(resolution, formula_size, oversized?)` so the caller can
/// surface telemetry signals.
fn smt_extreme(
    root: StmtId,
    stmts: &[IRStatement],
    cache: &HashMap<StmtId, ResolvedValue>,
    timeout_ms: u64,
    maximise: bool,
    max_formula_size: Option<usize>,
) -> (Option<i64>, usize, bool) {
    let mut walker = Walker::new(stmts, cache);
    walker.max_size = max_formula_size;
    let root_term = match walker.encode(root) {
        Some(t) => t,
        None => return (None, walker.visited, walker.aborted_oversized),
    };
    let formula_size = walker.visited;
    if walker.aborted_oversized {
        return (None, formula_size, true);
    }

    let opt = z3::Optimize::new();
    {
        let mut params = z3::Params::new();
        params.set_u32("timeout", timeout_ms.min(u32::MAX as u64) as u32);
        opt.set_params(&params);
    }
    for c in walker.constraints.drain(..) {
        opt.assert(&c);
    }

    // Structural-predicate integration for the `resolve_max` / `resolve_min`
    // path. Mirrors the one in `smt_query_with_budget` — without this, the
    // optimiser does not see the predicate-derived facts, so e.g. a query
    // for max(k) given `nnz(x) == k` and `len(x) == 1024` would return
    // unbounded.
    {
        use crate::optim::predicates::{
            build_input_array_lengths, build_input_name_index, Discharger,
        };
        use z3::ast::Int;

        let input_idx = build_input_name_index(stmts);
        let input_lengths = build_input_array_lengths(stmts);

        let mut name_to_int: HashMap<String, Int> = HashMap::new();
        for (name, producer_id) in &input_idx {
            if let Some(term) = walker.encode(*producer_id) {
                name_to_int.insert(name.clone(), term.as_int());
            }
        }

        let discharger = Discharger::new();
        let pred_c = discharger.collect_predicate_constraints(stmts, &input_lengths, |name| {
            if let Some(t) = name_to_int.get(name) {
                t.clone()
            } else if let Ok(n) = name.parse::<i64>() {
                Int::from_i64(n)
            } else {
                Int::fresh_const(&format!("sp_unknown_{name}_"))
            }
        });
        for c in &pred_c.clauses {
            opt.assert(c);
        }
        for f in &pred_c.meta_facts {
            opt.assert(f);
        }

        // Scalar-precondition discharge for the Optimize path (parallel
        // to the Solver path in smt_query_with_budget).
        use crate::optim::predicates::{find_scalar_preconditions, formula};
        for term_json in find_scalar_preconditions(stmts) {
            let term: formula::ContractTerm =
                match serde_json::from_str(term_json) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
            let mut subst = formula::Substitution::new();
            for (name, int_term) in &name_to_int {
                subst = subst.with_input(name.clone(), int_term.clone());
            }
            if let Ok(out) = formula::lower_bool(&term, &subst) {
                opt.assert(&out.term);
                for (_, facts) in &out.meta_fact_sets {
                    for f in facts {
                        opt.assert(f);
                    }
                }
            }
        }
    }

    let int = match root_term {
        crate::optim::smt_encoding::Z3Term::Int(i) => i,
        crate::optim::smt_encoding::Z3Term::Bool(b) => {
            // Project bool→int(0/1) so the Optimize objective makes sense.
            b.ite(&z3::ast::Int::from_i64(1), &z3::ast::Int::from_i64(0))
        }
        crate::optim::smt_encoding::Z3Term::Real(r) => {
            // Project real→int by floor — the integer-bounds path doesn't
            // return non-integer answers; downstream consumers expect i64.
            r.to_int()
        }
    };
    if maximise {
        opt.maximize(&int);
    } else {
        opt.minimize(&int);
    }
    let resolved = match opt.check(&[]) {
        z3::SatResult::Sat => {
            opt.get_model()
                .and_then(|model| model.eval(&int, true))
                .and_then(|v| v.as_i64())
        }
        z3::SatResult::Unsat | z3::SatResult::Unknown => None,
    };
    (resolved, formula_size, false)
}

/// Reverse-reachability walker. Translates IR → Z3 terms via the
/// `IROp::smt_encode` trait, threading a cache so cached resolutions
/// become literals (paper "encode cached resolutions as literals").
struct Walker<'a> {
    stmts: &'a [IRStatement],
    cache: &'a HashMap<StmtId, ResolvedValue>,
    encoded: HashMap<StmtId, crate::optim::smt_encoding::Z3Term>,
    constraints: Vec<z3::ast::Bool>,
    enc_ctx: crate::optim::smt_encoding::SmtEncodingCtx,
    /// P5 commit 3: optional cap on the number of distinct IR statements
    /// the walk visits before giving up. None = unbounded (legacy).
    max_size: Option<usize>,
    /// Number of statements the walker has *visited* so far this query
    /// (i.e., `encode(ptr)` calls that didn't short-circuit on the
    /// already-encoded or already-cached fast-paths). Counts pre-recursion
    /// so a deep chain trips the cap before unwinding finishes the
    /// inserts. Compared against `max_size`.
    visited: usize,
    /// Set when the walk hit `max_size` and aborted. The `encoded.len()`
    /// at that point is the snapshot we surface as `formula_size`.
    aborted_oversized: bool,
}

impl<'a> Walker<'a> {
    fn new(
        stmts: &'a [IRStatement],
        cache: &'a HashMap<StmtId, ResolvedValue>,
    ) -> Self {
        Self {
            stmts,
            cache,
            encoded: HashMap::new(),
            constraints: Vec::new(),
            enc_ctx: crate::optim::smt_encoding::SmtEncodingCtx::new(),
            max_size: None,
            visited: 0,
            aborted_oversized: false,
        }
    }

    /// Encode the wire at `ptr`, recursively encoding its dependencies.
    /// Returns the Z3 term that *represents* the wire's value. Each ptr
    /// is encoded at most once per query.
    fn encode(
        &mut self,
        ptr: StmtId,
    ) -> Option<crate::optim::smt_encoding::Z3Term> {
        if let Some(t) = self.encoded.get(&ptr) {
            return Some(t.clone());
        }

        // Cached resolution → emit a literal.
        if let Some(rv) = self.cache.get(&ptr) {
            let term = match rv {
                ResolvedValue::Int(n) => {
                    crate::optim::smt_encoding::Z3Term::Int(z3::ast::Int::from_i64(*n))
                }
                ResolvedValue::Bool(b) => {
                    crate::optim::smt_encoding::Z3Term::Bool(z3::ast::Bool::from_bool(*b))
                }
                ResolvedValue::Unknown => {
                    // No info; mint a fresh symbolic.
                    self.enc_ctx.fresh_unconstrained()
                }
            };
            self.encoded.insert(ptr, term.clone());
            return Some(term);
        }

        // Formula-size budget: abort early when we'd exceed the cap.
        // Counted at *visit time* (pre-recursion) so a deep chain trips
        // the cap before unwinding finishes the inserts; using
        // `encoded.len()` instead would only trip after all leaves have
        // returned, defeating the bound.
        self.visited += 1;
        if let Some(cap) = self.max_size {
            if self.visited > cap {
                self.aborted_oversized = true;
                return None;
            }
        }

        let stmt = self.stmts.get(ptr as usize)?;
        // Recurse on arguments.
        let mut arg_terms: Vec<crate::optim::smt_encoding::Z3Term> = Vec::new();
        for &arg in &stmt.arguments {
            arg_terms.push(self.encode(arg)?);
        }
        // Encode this op.
        use crate::optim::smt_encoding::IROp;
        let term = stmt.ir.smt_encode(&mut self.enc_ctx, &arg_terms);
        self.encoded.insert(ptr, term.clone());
        Some(term)
    }
}
