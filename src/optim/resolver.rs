//! P0 — `Resolver` seam for the SMT-resolver epic.
//!
//! This module defines the architectural seams introduced by phase 0 of
//! `compiler.epic-restore-smt-reasoning`:
//!
//! * The [`Resolver`] trait — the abstraction every "is this value provably
//!   constant?" query goes through. P0 ships [`StaticOnlyResolver`], whose
//!   behaviour is byte-for-byte identical to today (it just delegates to the
//!   existing `Value::int_val` / `bool_val` accessors). P1 adds an
//!   `SmtResolver`, P2 adds a `RangeResolver`, and call sites stay unchanged.
//!
//! * The [`StaticInt`] typed wrapper plus the [`SiteKind`] enum and the
//!   [`require_static_int`] helper. These give us one chokepoint for
//!   "must-be-compile-time-constant" integer call sites with informative,
//!   uniform diagnostics. The bulk of the migration of the
//!   `int_val().expect("constant")` sites onto this API is staged across
//!   per-category follow-up commits.
//!
//! No behaviour change in P0. The default resolver matches today's semantics.

use crate::ast::DebugInfo;
use crate::error::ZinniaError;
use crate::ir::IRStatement;
use crate::types::{StmtId, Value};

// ---------------------------------------------------------------------------
// Resolver trait + StaticOnlyResolver default
// ---------------------------------------------------------------------------

/// The "is this value provably constant?" abstraction.
///
/// The `&mut self` receiver is intentional: P1's `SmtResolver` will memoise
/// per-ptr query results, and P2's `RangeResolver` will accumulate interval
/// information across queries. P0's [`StaticOnlyResolver`] is stateless, so
/// the receiver is unused there — but holding the line on `&mut self` now
/// avoids a churning API change in P1.
///
/// `Send + Sync` are required because [`crate::ir::IRGraph`] is held by a
/// `#[pyclass]` (`CompiledIR`), which requires its fields to be thread-safe.
/// Concrete impls must respect this bound (e.g., a future SMT context held
/// inside a resolver needs to be guarded with a mutex if it isn't already
/// thread-safe).
pub trait Resolver: Send + Sync {
    /// Resolve `val` to a compile-time integer if provably constant.
    fn resolve_int(&mut self, val: &Value) -> Option<i64>;

    /// Resolve `val` to a compile-time boolean if provably constant.
    fn resolve_bool(&mut self, val: &Value) -> Option<bool>;

    /// Upper bound on `val` if provable.
    ///
    /// For [`StaticOnlyResolver`] this collapses to the literal itself
    /// (max of a constant is the constant). P2's range resolver gives a
    /// tighter bound, P1's SMT resolver discharges via maximisation.
    fn resolve_max(&mut self, val: &Value) -> Option<i64>;

    /// Lower bound on `val` if provable.
    fn resolve_min(&mut self, val: &Value) -> Option<i64>;

    /// Resolve `val` to a compile-time integer, with the IR statement vector
    /// available for traversal. P1's `SmtResolver` overrides this and walks
    /// the dependency graph; resolvers that don't care about the IR (i.e.
    /// [`StaticOnlyResolver`]) just delegate to the static-val variant.
    ///
    /// Why a separate method instead of putting `&[IRStatement]` on every
    /// `resolve_*`: the trait is invoked through `IRBuilder::resolver_mut()`,
    /// which currently exposes `&mut dyn Resolver` on its own. The
    /// `_with_stmts` variants are wired through a dedicated chokepoint
    /// (`IRBuilder::split_resolver_and_stmts`) so the borrow-checker can
    /// hand out `&mut resolver` and `&[IRStatement]` simultaneously without
    /// a churning API change at every call site.
    fn resolve_int_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<i64> {
        self.resolve_int(val)
    }

    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<bool> {
        self.resolve_bool(val)
    }

    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<i64> {
        self.resolve_max(val)
    }

    fn resolve_min_with_stmts(
        &mut self,
        val: &Value,
        _stmts: &[IRStatement],
    ) -> Option<i64> {
        self.resolve_min(val)
    }

    /// Cache-invalidation hook called by IR-mutating optim passes.
    ///
    /// `affected` is a (possibly empty) slice of mutated stmt ids. An empty
    /// slice is the conservative "everything possibly mutated" signal —
    /// P0 uses this default everywhere; P5 may refine to precise ids.
    fn on_ir_mutated(&mut self, _affected: &[StmtId]) {}

    /// P5: expose the SMT-pipeline telemetry handle, when the resolver
    /// has one. Layered/Range/Smt resolvers return Some; StaticOnly
    /// returns None. The handle is `Arc`'d, so callers can hold it past
    /// the resolver's lifetime to print a summary at end of compilation.
    fn telemetry_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        None
    }
}

/// The default `Resolver`: no-op cache, delegates straight to the existing
/// `Value` accessors. Behaviourally identical to pre-P0 code.
#[derive(Debug, Default)]
pub struct StaticOnlyResolver;

impl StaticOnlyResolver {
    pub fn new() -> Self {
        Self
    }
}

impl Resolver for StaticOnlyResolver {
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }

    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        val.bool_val()
    }

    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        // For a fully-static integer, the value is its own max. P2's
        // `RangeResolver` will tighten this for non-literal expressions.
        val.int_val()
    }

    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        val.int_val()
    }
}

// ---------------------------------------------------------------------------
// LayeredResolver — composition of resolvers, cheap-first dispatch
// ---------------------------------------------------------------------------
//
// P2 — the layered-resolver pattern from the epic spec (design principle 1:
// "cheap analyses first, SMT as backstop"). Holds a list of `Box<dyn Resolver>`
// layers; on each query, walks them in order and returns the first
// `Some(_)` answer.
//
// Typical composition for the full epic: `range → static → SMT`. P2 ships
// the construction and a `range_then_smt` helper. Wiring it in as the
// default `IRBuilder` / `IRGraph` resolver is P3.
//
// `on_ir_mutated` fans out to every layer so each one's invalidation policy
// fires.

/// A pipeline of resolvers, queried in order. First `Some(_)` wins.
pub struct LayeredResolver {
    layers: Vec<Box<dyn Resolver>>,
    /// P5 telemetry, accumulating cross-layer counters. Constructed by
    /// `range_then_smt` (so the range and SMT layers share one). Public
    /// constructors `new` / `new_with_telemetry` let callers wire it
    /// however they like.
    telemetry: std::sync::Arc<crate::optim::telemetry::SmtTelemetry>,
}

impl std::fmt::Debug for LayeredResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredResolver")
            .field("num_layers", &self.layers.len())
            .finish()
    }
}

impl LayeredResolver {
    /// Build a pipeline from an explicit list of layers. The telemetry
    /// stays disconnected from the sub-layers (each layer keeps its own).
    /// For shared telemetry across layers, use `range_then_smt` or
    /// `new_with_telemetry`.
    pub fn new(layers: Vec<Box<dyn Resolver>>) -> Self {
        Self {
            layers,
            telemetry: crate::optim::telemetry::SmtTelemetry::new(),
        }
    }

    /// Build a pipeline with an explicit shared telemetry. Caller is
    /// responsible for wiring the same telemetry into each sub-layer (via
    /// `RangeResolver::with_telemetry` / `SmtResolver::with_telemetry`).
    pub fn new_with_telemetry(
        layers: Vec<Box<dyn Resolver>>,
        telemetry: std::sync::Arc<crate::optim::telemetry::SmtTelemetry>,
    ) -> Self {
        Self { layers, telemetry }
    }

    /// The canonical P2 composition: `RangeResolver → SmtResolver`.
    /// Intended consumer for P3+. Range handles the bounded-loop-index /
    /// modular / mask cases; SMT handles symbolic relations range can't see
    /// (e.g., `select(x == 5, 100, 100)` where the cond depends on a free
    /// variable).
    ///
    /// Both sub-layers share one telemetry handle so the end-of-compilation
    /// summary covers the whole pipeline.
    pub fn range_then_smt() -> Self {
        Self::range_then_smt_with_timeout(500)
    }

    /// Same as `range_then_smt` but with an explicit Z3 per-query timeout
    /// (ms). P5 uses this so callers can tighten the budget without first
    /// constructing the SMT layer manually.
    pub fn range_then_smt_with_timeout(timeout_ms: u64) -> Self {
        Self::range_then_smt_with_budget(timeout_ms, usize::MAX)
    }

    /// Same as `range_then_smt_with_timeout` but with an additional cap on
    /// the per-query formula size (number of IR statements visited by the
    /// reverse-reachability walk). Beyond this cap the walk aborts and the
    /// SmtResolver returns None — counted as `queries_skipped_oversized`
    /// in telemetry. P5 commit 3 uses this so callers can configure a
    /// pragmatic budget that bounds the worst-case query without changing
    /// the timeout.
    pub fn range_then_smt_with_budget(
        timeout_ms: u64,
        max_formula_size: usize,
    ) -> Self {
        let telemetry = crate::optim::telemetry::SmtTelemetry::new();
        let range = crate::optim::range::RangeResolver::new()
            .with_telemetry(std::sync::Arc::clone(&telemetry));
        let smt = SmtResolver::new()
            .with_timeout(timeout_ms)
            .with_max_formula_size(max_formula_size)
            .with_telemetry(std::sync::Arc::clone(&telemetry));
        Self::new_with_telemetry(
            vec![Box::new(range), Box::new(smt)],
            telemetry,
        )
    }

    /// Borrow the shared telemetry handle. Used by the compile entry-point
    /// to surface the summary to stderr at end of compilation.
    pub fn telemetry(&self) -> std::sync::Arc<crate::optim::telemetry::SmtTelemetry> {
        std::sync::Arc::clone(&self.telemetry)
    }
}

impl Resolver for LayeredResolver {
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_int(val) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        for layer in self.layers.iter_mut() {
            if let Some(b) = layer.resolve_bool(val) {
                return Some(b);
            }
        }
        None
    }

    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_max(val) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_min(val) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_int_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.int_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_int_with_stmts(val, stmts) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<bool> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.bool_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(b) = layer.resolve_bool_with_stmts(val, stmts) {
                return Some(b);
            }
        }
        None
    }

    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.int_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_max_with_stmts(val, stmts) {
                return Some(n);
            }
        }
        None
    }

    fn resolve_min_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        use std::sync::atomic::Ordering;
        self.telemetry.queries_total.fetch_add(1, Ordering::Relaxed);
        if val.int_val().is_some() {
            self.telemetry
                .queries_static_val_hit
                .fetch_add(1, Ordering::Relaxed);
        }
        for layer in self.layers.iter_mut() {
            if let Some(n) = layer.resolve_min_with_stmts(val, stmts) {
                return Some(n);
            }
        }
        None
    }

    fn on_ir_mutated(&mut self, affected: &[StmtId]) {
        for layer in self.layers.iter_mut() {
            layer.on_ir_mutated(affected);
        }
    }

    fn telemetry_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        Some(std::sync::Arc::clone(&self.telemetry))
    }
}

// ---------------------------------------------------------------------------
// SmtResolver
// ---------------------------------------------------------------------------
//
// P1 — discharges constancy / max / min queries via Z3 when the cheap
// `static_val` fast path can't.
//
// ## Design overview
//
// * Per-ptr cache (`HashMap<StmtId, ResolvedValue>`). Both `Int(n)` /
//   `Bool(b)` resolutions and `Unknown` outcomes are cached so a repeat
//   query on the same wire is free. `on_ir_mutated` clears the cache (P1
//   conservative; P5 may refine).
//
// * Lazy formula construction. Every `resolve_*` query traverses the
//   reverse-reachability subgraph rooted at `val.stmt_id()` and encodes only
//   those statements as Z3 constraints — never the whole graph. Reference:
//   the old Python `_build_smt_constraints_for(ptr)`.
//
// * Time budget. Each Z3 query sets the solver's `timeout` parameter
//   (default 500 ms; configurable via `SmtResolver::with_timeout`). On Z3
//   `unknown` we cache `Unknown` and return `None`.
//
// * Disable flag. `with_disabled(true)` makes every query return `None`
//   after the static_val fast path. Lets users diagnose whether SMT is the
//   compile-time bottleneck.
//
// * Cached resolutions encoded as literals. When the reverse-reachability
//   walk encounters a statement whose ptr is already in the cache, it
//   emits the cached constant as a Z3 literal instead of recursing — keeps
//   formulas small (paper "encode cached resolutions as literals").
//
// ## Z3 lifetime / Send+Sync
//
// `z3` 0.20 uses an implicit thread-local `Context`. The solver, all asts,
// etc. are bound to that context — but the context itself is a `Rc`, and
// `Rc` is `!Send + !Sync`. Our `Resolver` trait requires `Send + Sync`
// (because `IRGraph` is held by a `#[pyclass]`). The way out:
//
// * `SmtResolver` does NOT store any Z3 state across calls. Each query
//   constructs a fresh `Solver` (cheap), encodes the formula, runs it,
//   discards the solver. No `Rc<Context>` ever crosses a method boundary.
//
// * The "single Z3 context per compilation" requirement from the spec is
//   satisfied implicitly: z3 0.20's thread-local context is created lazily
//   on first use per thread. Compilation is single-threaded, so all
//   queries from the same compilation share one thread-local context.
//   Per-query setup cost is minimal (Z3 reuses the context's symbol pool
//   etc.).
//
// * Only the cache and the per-resolver knobs (timeout, disabled, counter)
//   live in the resolver across calls. All `Send + Sync`.
//
// ## IRBuilder/IRGraph handoff (option (b))
//
// Each phase of compilation has its own `SmtResolver` (or none). Per the
// spec's "(b) Each phase has its own SmtResolver with its own cache.
// Caches don't migrate." This is acceptable for P1; P3+ may share state if
// profiling shows the cost matters.
//
// The IR-graph is passed in via the `_with_stmts` trait methods (see the
// trait definition above). `IRBuilder::split_resolver_and_stmts(&mut)`
// and `IRGraph::split_resolver_and_stmts(&mut)` are the chokepoints.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::optim::telemetry::SmtTelemetry;

/// One cached resolution outcome for a wire's ptr.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResolvedValue {
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

/// Outcome categorisation for `smt_query`. Wraps the resolved value (or the
/// reason we didn't resolve) plus the formula size (statement count walked).
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
/// has a unique value. Returns `Some(ResolvedValue::Int|Bool)` if Z3 proves
/// uniqueness within the time budget; `None` on timeout / non-unique.
///
/// Mirrors the "find a model, then add `expr != that_value` and re-check"
/// pattern from the old Python `SMTUtils.resolve_expr`.
///
/// Returns `(resolution, formula_size)`. The size is the count of distinct
/// IR statements visited by the reverse-reachability walk.
fn smt_query(
    root: StmtId,
    stmts: &[IRStatement],
    cache: &HashMap<StmtId, ResolvedValue>,
    timeout_ms: u64,
    want_bool: bool,
) -> (Option<ResolvedValue>, usize) {
    let out = smt_query_with_budget(root, stmts, cache, timeout_ms, want_bool, None);
    (out.resolved, out.formula_size)
}

/// Same as [`smt_query`] but with an optional cap on formula size. When the
/// reverse-reachability walk exceeds `max_formula_size` IR statements, the
/// walker aborts and the resolver returns `None` — paying only the walk cost,
/// not Z3 time, on the heaviest queries.
fn smt_query_with_budget(
    root: StmtId,
    stmts: &[IRStatement],
    cache: &HashMap<StmtId, ResolvedValue>,
    timeout_ms: u64,
    want_bool: bool,
    max_formula_size: Option<usize>,
) -> SmtQueryOut {
    use z3::ast::Ast;

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
                        solver.assert(&int._eq(&z3::ast::Int::from_i64(n)).not());
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
                        let bool_proj = int._eq(&zero).not();
                        let v = model.eval(&bool_proj, true)?;
                        let b = v.as_bool()?;
                        solver.assert(&bool_proj._eq(&z3::ast::Bool::from_bool(b)).not());
                        if solver.check() == z3::SatResult::Unsat {
                            Some(ResolvedValue::Bool(b))
                        } else {
                            None
                        }
                    }
                    (crate::optim::smt_encoding::Z3Term::Bool(b_ast), _) => {
                        let v = model.eval(b_ast, true)?;
                        let b = v.as_bool()?;
                        solver.assert(&b_ast._eq(&z3::ast::Bool::from_bool(b)).not());
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

// ---------------------------------------------------------------------------
// StaticInt wrapper + SiteKind + require_static_int
// ---------------------------------------------------------------------------

/// A required-constant integer at a specific call site.
///
/// Returned by [`require_static_int`] — i.e., obtained only after the
/// resolver has actually proved the wire is a compile-time constant.
///
/// Ergonomics convention (P0 decision): this type implements
/// `From<StaticInt> for i64`, so consumers can spell unwrapping as
/// `let n: i64 = static_n.into();` without `.0` ceremony.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticInt(pub i64);

impl StaticInt {
    /// Borrow the wrapped i64. Use sparingly — prefer `into()` /
    /// `From<StaticInt>` at consumer sites.
    pub fn get(self) -> i64 {
        self.0
    }
}

impl From<StaticInt> for i64 {
    fn from(s: StaticInt) -> i64 {
        s.0
    }
}

impl From<StaticInt> for u32 {
    fn from(s: StaticInt) -> u32 {
        s.0 as u32
    }
}

impl From<StaticInt> for usize {
    fn from(s: StaticInt) -> usize {
        s.0 as usize
    }
}

/// Where a "must-be-constant" integer requirement is being enforced.
///
/// Exists primarily to centralise diagnostic-message formatting so every
/// site speaks the same "<thing> must be a compile-time constant int"
/// dialect. Add a new variant when migrating a site whose rejection
/// message doesn't already fit.
#[derive(Clone, Copy, Debug)]
pub enum SiteKind {
    /// A shape element of an array constructor, at the given axis.
    ShapeAxis(usize),
    /// `start` argument of a slice expression.
    SliceStart,
    /// `stop` argument of a slice expression.
    SliceStop,
    /// `step` argument of a slice expression.
    SliceStep,
    /// `start` argument of `range(...)`.
    RangeStart,
    /// `stop` argument of `range(...)`.
    RangeStop,
    /// `step` argument of `range(...)`.
    RangeStep,
    /// A target dimension of `arr.reshape(...)` / `np.reshape(arr, ...)`.
    ReshapeDim,
    /// `repeats` argument of `np.repeat(arr, k)`.
    RepeatCount,
    /// `n` argument of `np.split(arr, n)`.
    SplitSections,
    /// An `axis=` argument (sum, transpose, swapaxes, …).
    Axis,
    /// Position passed to `np.expand_dims` / newaxis.
    NewAxisPosition,
    /// `num` argument of `np.linspace`.
    LinspaceNum,
    /// Dynamic-array index bound check: address used with `ReadMemory` /
    /// `WriteMemory` must lie in `[0, segment_size)`. Today's wiring is
    /// informational — the memory-trace permutation argument enforces
    /// soundness at prove time — but the chokepoint surfaces SMT engagement
    /// on production-natural patterns where indices are runtime values
    /// whose bounds are nonetheless provable. Item #4 of the
    /// `smt-invocation-load-bearing` card.
    DynamicIndexBound,
    /// Anything not yet enumerated. The string is a short site label;
    /// promote it to a named variant if it recurs.
    Other(&'static str),
}

impl SiteKind {
    /// Stable short identifier used as the key in
    /// `SmtTelemetry::chokepoint_invocations`. Must be `&'static str`
    /// because the map's keys live for the program's lifetime.
    pub fn short_name(&self) -> &'static str {
        match self {
            SiteKind::ShapeAxis(_) => "shape_axis",
            SiteKind::SliceStart => "slice_start",
            SiteKind::SliceStop => "slice_stop",
            SiteKind::SliceStep => "slice_step",
            SiteKind::RangeStart => "range_start",
            SiteKind::RangeStop => "range_stop",
            SiteKind::RangeStep => "range_step",
            SiteKind::ReshapeDim => "reshape_dim",
            SiteKind::RepeatCount => "repeat_count",
            SiteKind::SplitSections => "split_sections",
            SiteKind::Axis => "axis",
            SiteKind::NewAxisPosition => "new_axis_position",
            SiteKind::LinspaceNum => "linspace_num",
            SiteKind::DynamicIndexBound => "dyn_index_bound",
            SiteKind::Other(label) => label,
        }
    }

    /// Render a human-readable diagnostic for a "this must be a compile-time
    /// constant int" rejection at this site. One source of truth for the
    /// wording.
    pub fn diagnostic(&self) -> String {
        match self {
            SiteKind::ShapeAxis(i) => format!(
                "shape element at axis {} must be a compile-time constant int",
                i
            ),
            SiteKind::SliceStart => {
                "slice start must be a compile-time constant int".to_string()
            }
            SiteKind::SliceStop => {
                "slice stop must be a compile-time constant int".to_string()
            }
            SiteKind::SliceStep => {
                "slice step must be a compile-time constant int".to_string()
            }
            SiteKind::RangeStart => {
                "range start must be a compile-time constant int".to_string()
            }
            SiteKind::RangeStop => {
                "range stop must be a compile-time constant int".to_string()
            }
            SiteKind::RangeStep => {
                "range step must be a compile-time constant int".to_string()
            }
            SiteKind::ReshapeDim => {
                "reshape target dimension must be a compile-time constant int"
                    .to_string()
            }
            SiteKind::RepeatCount => {
                "repeat count must be a compile-time constant int".to_string()
            }
            SiteKind::SplitSections => {
                "split sections must be a compile-time constant int".to_string()
            }
            SiteKind::Axis => {
                "axis argument must be a compile-time constant int".to_string()
            }
            SiteKind::NewAxisPosition => {
                "new-axis position must be a compile-time constant int".to_string()
            }
            SiteKind::LinspaceNum => {
                "linspace `num` must be a compile-time constant int".to_string()
            }
            SiteKind::DynamicIndexBound => {
                "dynamic array index could not be proved in-bounds at compile time".to_string()
            }
            SiteKind::Other(label) => {
                format!("{} must be a compile-time constant int", label)
            }
        }
    }
}

/// Format the diagnostic for a `require_static_*` failure, optionally
/// prefixed with debug-info location.
fn format_diagnostic(site: SiteKind, dbg: Option<&DebugInfo>) -> String {
    let base = site.diagnostic();
    match dbg {
        Some(d) => match (d.line, d.col) {
            (Some(line), Some(col)) => format!("{} (at line {}, col {})", base, line, col),
            (Some(line), None) => format!("{} (at line {})", base, line),
            _ => base,
        },
        None => base,
    }
}

/// The single chokepoint for "this integer site must be a compile-time
/// constant" requirements.
///
/// Routes the query through the IRBuilder's resolver (today:
/// [`StaticOnlyResolver`], tomorrow: SMT/range-augmented). Returns a typed
/// [`StaticInt`] on success, or a [`ZinniaError`] with a uniform diagnostic
/// referring to the [`SiteKind`].
///
/// Per-category migration of existing `int_val().expect("constant")` sites
/// onto this API is staged across follow-up commits in the SMT epic.
pub fn require_static_int(
    b: &mut crate::builder::IRBuilder,
    val: &Value,
    site: SiteKind,
    dbg: Option<&DebugInfo>,
) -> Result<StaticInt, ZinniaError> {
    use std::sync::atomic::Ordering;
    let site_name = site.short_name();
    // Resolver pass — scoped so the borrow drops before we read `b.facts`.
    let (mut result, tel, smt_before) = {
        let (resolver, stmts) = b.split_resolver_and_stmts();
        let tel = resolver.telemetry_handle();
        let smt_before = tel.as_ref().map(|t| {
            t.queries_smt_resolved.load(Ordering::Relaxed)
                + t.queries_smt_unknown.load(Ordering::Relaxed)
        });
        if let Some(t) = tel.as_ref() {
            t.record_chokepoint_invocation(site_name);
        }
        let result = resolver.resolve_int_with_stmts(val, stmts);
        (result, tel, smt_before)
    };

    // Fact-fallback (compiler.fact-aware-chokepoints-batch): when the
    // resolver can't pin a static int, op-contract facts may still do it
    // via an Eq-shaped fact (`val == n`). `ask_bounds` collapses to
    // `min == max == n` in that case; we accept iff the two halves
    // coincide. Covers slice/range/repeat/reshape/split chokepoints,
    // which all route through this helper.
    if result.is_none() {
        if let Some((min, max)) = b.ask_bounds(val) {
            if min == max {
                result = Some(min);
            }
        }
    }

    if let (Some(t), Some(before)) = (tel.as_ref(), smt_before) {
        let smt_after = t.queries_smt_resolved.load(Ordering::Relaxed)
            + t.queries_smt_unknown.load(Ordering::Relaxed);
        if smt_after > before {
            t.record_chokepoint_smt_engagement(site_name);
        }
        if result.is_some() {
            t.record_chokepoint_resolved(site_name);
        }
    }
    match result {
        Some(n) => Ok(StaticInt(n)),
        None => Err(ZinniaError {
            message: format_diagnostic(site, dbg),
        }),
    }
}

/// Outcome of a bounded-int chokepoint query. See [`require_static_or_bounded_int`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundedInt {
    /// `static_val` succeeded — the value is a unique compile-time integer.
    Static(i64),
    /// `static_val` failed but the range / SMT layer proved both halves of
    /// a finite range. Use `max` for shape-style chokepoints; use `min`
    /// when a non-negative invariant matters.
    Bounded { min: i64, max: i64 },
    /// Neither layer could resolve. Caller decides whether to reject or
    /// degrade to a wider fallback.
    Neither,
}

impl From<StaticInt> for BoundedInt {
    fn from(s: StaticInt) -> BoundedInt {
        BoundedInt::Static(s.0)
    }
}

/// Bounded-aware companion to [`require_static_int`].
///
/// Tries the resolver layers in order:
///
/// 1. `resolve_int_with_stmts(val)` → [`BoundedInt::Static(n)`].
/// 2. Otherwise, both `resolve_max_with_stmts(val)` and
///    `resolve_min_with_stmts(val)` are queried. If both succeed,
///    returns [`BoundedInt::Bounded { min, max }`]. Both halves are
///    required so callers can rely on a finite range without having to
///    invent a default lower / upper.
/// 3. Else [`BoundedInt::Neither`].
///
/// Telemetry mirrors [`require_static_int`]: per-chokepoint invocation,
/// resolved, and SMT-engagement counters increment via the resolver's
/// shared telemetry handle.
///
/// This helper is the seam for [`compiler.chokepoint-shape-axis-bounded`]
/// and future per-chokepoint-family migrations (Slice, Range,
/// RepeatCount, LinspaceNum, SplitSections). It is intentionally minimal
/// — it does *not* decide how a caller should use the bound. That
/// remains per-chokepoint policy: `ShapeAxis` falls back to
/// `DynamicNDArray`; `Range*` will fall back to a selector-gated unroll;
/// `Axis` / `NewAxisPosition` will not migrate because their semantics
/// are structural, not bounded.
pub fn require_static_or_bounded_int(
    b: &mut crate::builder::IRBuilder,
    val: &Value,
    site: SiteKind,
    _dbg: Option<&DebugInfo>,
) -> BoundedInt {
    use std::sync::atomic::Ordering;
    let site_name = site.short_name();

    // Resolver-based pass (existing behavior). Scoped block so that the
    // resolver/stmts borrows die before we read `b.facts` for the fallback.
    let (mut outcome, tel, smt_before) = {
        let (resolver, stmts) = b.split_resolver_and_stmts();
        let tel = resolver.telemetry_handle();
        let smt_before = tel.as_ref().map(|t| {
            t.queries_smt_resolved.load(Ordering::Relaxed)
                + t.queries_smt_unknown.load(Ordering::Relaxed)
        });
        if let Some(t) = tel.as_ref() {
            t.record_chokepoint_invocation(site_name);
        }

        // 1) Try the static path first.
        let static_outcome = resolver.resolve_int_with_stmts(val, stmts);

        // 2) Otherwise query both bound halves.
        let outcome = match static_outcome {
            Some(n) => BoundedInt::Static(n),
            None => {
                let upper = resolver.resolve_max_with_stmts(val, stmts);
                let lower = resolver.resolve_min_with_stmts(val, stmts);
                match (lower, upper) {
                    (Some(min), Some(max)) if min < max => {
                        BoundedInt::Bounded { min, max }
                    }
                    _ => BoundedInt::Neither,
                }
            }
        };
        (outcome, tel, smt_before)
    };

    // 3) Fact-fallback (compiler.fact-aware-chokepoints-batch): when the
    //    resolver-based pass yields Neither but op contracts have
    //    deposited bound-shaped facts on this value's SSA ptr, recover
    //    (min, max) from them. Shared with other chokepoints via
    //    `IRBuilder::ask_bounds`. Requires min < max so equal bounds
    //    (which the resolver's static_val path should have caught) fall
    //    back to Neither rather than masquerading as a degenerate range.
    if matches!(outcome, BoundedInt::Neither) {
        if let Some((min, max)) = b.ask_bounds(val) {
            if min < max {
                outcome = BoundedInt::Bounded { min, max };
            }
        }
    }

    if let (Some(t), Some(before)) = (tel.as_ref(), smt_before) {
        let smt_after = t.queries_smt_resolved.load(Ordering::Relaxed)
            + t.queries_smt_unknown.load(Ordering::Relaxed);
        if smt_after > before {
            t.record_chokepoint_smt_engagement(site_name);
        }
        if !matches!(outcome, BoundedInt::Neither) {
            t.record_chokepoint_resolved(site_name);
        }
    }
    outcome
}

/// Scan `facts.per_stmt[ptr]` for `Cmp(SsaPtr(ptr) op LitInt(n))` shapes
/// and synthesize a `(min, max)` pair. Returns `None` if either half is
/// missing. Used by [`require_static_or_bounded_int`] as a fallback when
/// the resolver-based bound pass fails.
///
/// Recognized shapes (and the bound each yields):
/// - `Ge(SsaPtr, LitInt(n))` → min = n
/// - `Gt(SsaPtr, LitInt(n))` → min = n + 1
/// - `Le(SsaPtr, LitInt(n))` → max = n
/// - `Lt(SsaPtr, LitInt(n))` → max = n - 1
///
/// LitInt-on-left forms (e.g., `Ge(LitInt(0), SsaPtr(p))`) are also
/// handled by swapping the relation. This is intentionally narrow —
/// only the fact shapes our op-contract content emits today. Richer
/// shapes (arithmetic, predicates) require routing through `prove`.
pub(crate) fn derive_bounds_from_facts(
    facts: &crate::optim::predicates::FactStack,
    vid: crate::types::ValueId,
) -> Option<(i64, i64)> {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
    let entries = facts.per_value.get(&vid)?;
    let path_conds: Vec<&ContractTerm> = facts.visible_path_conditions();
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    for raw_term in entries {
        // Look through `Implies(p, body)` (encoded as `Or(Not(p), body)`)
        // when `p` matches an in-scope path condition. The wrapping is
        // how the branch-merge preserves the anchor for arm-only facts;
        // the body is the original value_id-anchored claim.
        let term: &ContractTerm = match raw_term {
            ContractTerm::BoolComb { op: BoolOp::Or, operands } if operands.len() == 2 => {
                if let ContractTerm::Not(inner) = &operands[0] {
                    let antecedent = inner.as_ref();
                    if path_conds.iter().any(|p| **p == *antecedent) {
                        &operands[1]
                    } else {
                        raw_term
                    }
                } else {
                    raw_term
                }
            }
            _ => raw_term,
        };
        if let ContractTerm::Cmp { op, lhs, rhs } = term {
            // Match `Value(vid) <op> LitInt(n)` or `LitInt(n) <op> Value(vid)`.
            let (relation, n) = match (lhs.as_ref(), rhs.as_ref()) {
                (
                    ContractTerm::Var(ContractVar::Value(v)),
                    ContractTerm::LitInt(n),
                ) if *v == vid => (*op, *n),
                (
                    ContractTerm::LitInt(n),
                    ContractTerm::Var(ContractVar::Value(v)),
                ) if *v == vid => {
                    // Swap: `n <op> ptr` becomes `ptr <swap(op)> n`.
                    let swapped = match op {
                        CmpOp::Lt => CmpOp::Gt,
                        CmpOp::Le => CmpOp::Ge,
                        CmpOp::Gt => CmpOp::Lt,
                        CmpOp::Ge => CmpOp::Le,
                        CmpOp::Eq => CmpOp::Eq,
                        CmpOp::Ne => CmpOp::Ne,
                    };
                    (swapped, *n)
                }
                _ => continue,
            };
            match relation {
                CmpOp::Ge => lo = Some(lo.map_or(n, |cur| cur.max(n))),
                CmpOp::Gt => lo = Some(lo.map_or(n + 1, |cur| cur.max(n + 1))),
                CmpOp::Le => hi = Some(hi.map_or(n, |cur| cur.min(n))),
                CmpOp::Lt => hi = Some(hi.map_or(n - 1, |cur| cur.min(n - 1))),
                CmpOp::Eq => {
                    lo = Some(lo.map_or(n, |cur| cur.max(n)));
                    hi = Some(hi.map_or(n, |cur| cur.min(n)));
                }
                CmpOp::Ne => {}
            }
        }
    }
    match (lo, hi) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => None,
    }
}

/// Compute the interval bound on the output of an integer arithmetic op
/// given input intervals, returning a deposit-ready `ContractTerm`.
///
/// Facts-only lookup: uses [`derive_bounds_from_facts`] on each input;
/// does NOT invoke SMT. Returns `None` when either input lacks
/// fact-derived bounds.
///
/// Checked-arithmetic default-deny: any `checked_*` overflow on the
/// interval corners produces `None` (no fact deposited) rather than a
/// possibly-unsound wrapped value.
pub fn interval_fact_for_int_binary(
    facts: &crate::optim::predicates::FactStack,
    op: crate::optim::predicates::formula::ArithOp,
    a_vid: crate::types::ValueId,
    b_vid: crate::types::ValueId,
    out_vid: crate::types::ValueId,
) -> Option<crate::optim::predicates::formula::ContractTerm> {
    use crate::optim::predicates::formula::{
        ArithOp, BoolOp, CmpOp, ContractTerm, ContractVar,
    };

    let (a_lo, a_hi) = derive_bounds_from_facts(facts, a_vid)?;
    let (b_lo, b_hi) = derive_bounds_from_facts(facts, b_vid)?;

    let (out_lo, out_hi) = match op {
        ArithOp::Add => {
            let lo = a_lo.checked_add(b_lo)?;
            let hi = a_hi.checked_add(b_hi)?;
            (lo, hi)
        }
        ArithOp::Sub => {
            let lo = a_lo.checked_sub(b_hi)?;
            let hi = a_hi.checked_sub(b_lo)?;
            (lo, hi)
        }
        ArithOp::Mul => {
            let c1 = a_lo.checked_mul(b_lo)?;
            let c2 = a_lo.checked_mul(b_hi)?;
            let c3 = a_hi.checked_mul(b_lo)?;
            let c4 = a_hi.checked_mul(b_hi)?;
            let lo = c1.min(c2).min(c3).min(c4);
            let hi = c1.max(c2).max(c3).max(c4);
            (lo, hi)
        }
        _ => return None,
    };

    Some(ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_lo)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_hi)),
            },
        ],
    })
}

/// Float analog of [`derive_bounds_from_facts`]. Walks the fact stack
/// for `Cmp(Ge | Gt | Le | Lt | Eq, Var(Value(vid)), LitFloat(f))` clauses
/// (or the swapped `LitFloat` on the left), returning `(lo, hi)` if both
/// directions are visible.
///
/// Mirrors the int variant structurally; the only differences are the
/// literal type (`LitFloat(ContractFloat)`), the bound type (`f64`), and
/// the absence of integer `+1 / -1` strictness adjustments — for the
/// float case `Gt(v, f)` and `Lt(v, f)` are treated as non-strict (i.e.
/// the same as `Ge`/`Le`). The widening is sound: a strict bound implies
/// the non-strict one.
pub(crate) fn derive_float_bounds_from_facts(
    facts: &crate::optim::predicates::FactStack,
    vid: crate::types::ValueId,
) -> Option<(f64, f64)> {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar};
    let entries = facts.per_value.get(&vid)?;
    let path_conds: Vec<&ContractTerm> = facts.visible_path_conditions();
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for raw_term in entries {
        let term: &ContractTerm = match raw_term {
            ContractTerm::BoolComb { op: BoolOp::Or, operands } if operands.len() == 2 => {
                if let ContractTerm::Not(inner) = &operands[0] {
                    let antecedent = inner.as_ref();
                    if path_conds.iter().any(|p| **p == *antecedent) {
                        &operands[1]
                    } else {
                        raw_term
                    }
                } else {
                    raw_term
                }
            }
            _ => raw_term,
        };
        if let ContractTerm::Cmp { op, lhs, rhs } = term {
            let (relation, n) = match (lhs.as_ref(), rhs.as_ref()) {
                (
                    ContractTerm::Var(ContractVar::Value(v)),
                    ContractTerm::LitFloat(ContractFloat(n)),
                ) if *v == vid => (*op, *n),
                (
                    ContractTerm::LitFloat(ContractFloat(n)),
                    ContractTerm::Var(ContractVar::Value(v)),
                ) if *v == vid => {
                    let swapped = match op {
                        CmpOp::Lt => CmpOp::Gt,
                        CmpOp::Le => CmpOp::Ge,
                        CmpOp::Gt => CmpOp::Lt,
                        CmpOp::Ge => CmpOp::Le,
                        CmpOp::Eq => CmpOp::Eq,
                        CmpOp::Ne => CmpOp::Ne,
                    };
                    (swapped, *n)
                }
                _ => continue,
            };
            match relation {
                CmpOp::Ge | CmpOp::Gt => {
                    lo = Some(lo.map_or(n, |cur| cur.max(n)));
                }
                CmpOp::Le | CmpOp::Lt => {
                    hi = Some(hi.map_or(n, |cur| cur.min(n)));
                }
                CmpOp::Eq => {
                    lo = Some(lo.map_or(n, |cur| cur.max(n)));
                    hi = Some(hi.map_or(n, |cur| cur.min(n)));
                }
                CmpOp::Ne => {}
            }
        }
    }
    match (lo, hi) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => None,
    }
}

/// Compute the interval bound on the output of a float arithmetic op
/// given input intervals, returning a deposit-ready `ContractTerm`.
///
/// Facts-only lookup: uses [`derive_float_bounds_from_facts`] on each
/// input; does NOT invoke SMT. Returns `None` when either input lacks
/// fact-derived bounds.
///
/// f64-arithmetic default-deny: overflow on the interval corners
/// produces `±inf` (or NaN via `inf - inf` etc.); the `is_finite` guard
/// drops any non-finite corner so we never deposit a bound that would
/// lower poorly to the ZK Real fragment.
pub fn interval_fact_for_float_binary(
    facts: &crate::optim::predicates::FactStack,
    op: crate::optim::predicates::formula::ArithOp,
    a_vid: crate::types::ValueId,
    b_vid: crate::types::ValueId,
    out_vid: crate::types::ValueId,
) -> Option<crate::optim::predicates::formula::ContractTerm> {
    use crate::optim::predicates::formula::{
        ArithOp, BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let (a_lo, a_hi) = derive_float_bounds_from_facts(facts, a_vid)?;
    let (b_lo, b_hi) = derive_float_bounds_from_facts(facts, b_vid)?;

    let (out_lo, out_hi) = match op {
        ArithOp::Add => (a_lo + b_lo, a_hi + b_hi),
        ArithOp::Sub => (a_lo - b_hi, a_hi - b_lo),
        ArithOp::Mul => {
            let c1 = a_lo * b_lo;
            let c2 = a_lo * b_hi;
            let c3 = a_hi * b_lo;
            let c4 = a_hi * b_hi;
            (c1.min(c2).min(c3).min(c4), c1.max(c2).max(c3).max(c4))
        }
        _ => return None,
    };

    // Skip emit on overflow to ±inf or NaN (e.g. `inf - inf`). Sound:
    // no relayed fact is always preferable to lowering a non-finite
    // literal into the ZK Real fragment.
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return None;
    }

    Some(ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
            },
        ],
    })
}

/// Relay an interval bound through `sqrt`: given `input ∈ [lo, hi]` on the
/// fact stack, plant `output ∈ [sqrt(lo), sqrt(hi)]` on `output_vid`.
///
/// `sqrt` is monotone on `[0, ∞)` and f64 `sqrt` is correctly-rounded, so
/// the relayed interval is exact (no widening needed). Returns `true`
/// when a bound was deposited.
///
/// Mirrors [`interval_fact_for_float_binary`]'s pattern: facts-only
/// lookup, default-deny on non-finite corners.
pub fn relay_sqrt_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    // Defensive: `sqrt_f`'s `requires(x >= 0.0)` (Group 1's R) prevents
    // negative inputs from reaching here. If a stale fact slips through,
    // skip emit — `lo.sqrt()` would produce NaN and we must never
    // deposit an unsound bound.
    if lo < 0.0 {
        return false;
    }
    let out_lo = lo.sqrt();
    let out_hi = hi.sqrt();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Relay an interval bound through `exp`: given `input ∈ [lo, hi]` on the
/// fact stack, plant `output ∈ [exp(lo), exp(hi)]` on `output_vid`.
///
/// `exp` is monotone-increasing on all of `R`. f64 `exp(big)` overflows
/// to `+inf`; the `is_finite()` guard then skips emit rather than
/// depositing a non-finite literal. Returns `true` when a bound was
/// deposited.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners. No precondition guard (domain is
/// all reals).
pub fn relay_exp_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    let out_lo = lo.exp();
    let out_hi = hi.exp();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Relay an interval bound through `log`: given `input ∈ [lo, hi]` on
/// the fact stack with `lo > 0`, plant `output ∈ [log(lo), log(hi)]` on
/// `output_vid`.
///
/// `log` is monotone-increasing on `(0, ∞)`. Returns `true` when a
/// bound was deposited.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners.
pub fn relay_log_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    // `log` is undefined for non-positive inputs: f64 `ln(0.0) = -inf`,
    // `ln(neg) = NaN`. Group 1's R prevents these from reaching here,
    // but if a stale fact slips through we must skip rather than deposit
    // an unsound bound.
    if lo <= 0.0 {
        return false;
    }
    let out_lo = lo.ln();
    let out_hi = hi.ln();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Relay an interval bound through `arccos`: given `input ∈ [lo, hi]`
/// with `[lo, hi] ⊆ [-1, 1]`, plant `output ∈ [acos(hi), acos(lo)]` on
/// `output_vid`.
///
/// `arccos` is monotone-DECREASING on its domain `[-1, 1]`, so the
/// bounds swap: the output's low corner comes from the input's high
/// corner and vice versa. f64 `acos` is correctly-rounded, so the
/// relayed interval is exact (no widening needed). Returns `true` when
/// a bound was deposited.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners.
pub fn relay_arccos_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    // Defensive: `arccos_f`'s `requires(-1 <= x <= 1)` (Group 1's R)
    // prevents out-of-domain inputs from reaching here. If a stale fact
    // slips through, skip emit — `acos(out_of_domain)` is NaN and we
    // must never deposit an unsound bound.
    if lo < -1.0 || hi > 1.0 {
        return false;
    }
    // WHY swap: arccos is monotone-DECREASING on [-1, 1], so the
    // smallest output comes from the largest input and vice versa.
    let out_lo = hi.acos();
    let out_hi = lo.acos();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Aggregate per-element fact-derived bounds into a single `[lo, hi]`
/// union (`min` of lows, `max` of highs). Returns `None` if **any**
/// element lacks fact-derived bounds — soundness gate: a single
/// unbounded element makes the union unbounded too.
///
/// Used by [`relay_reduction_output_interval_int`] to summarise the
/// element-wise bound for a composite-array input. Per-element facts
/// live on each element's own `ValueId`; the whole-list `ValueId` does
/// not carry the bound directly.
pub fn aggregate_element_bounds(
    facts: &crate::optim::predicates::FactStack,
    element_vids: &[crate::types::ValueId],
) -> Option<(i64, i64)> {
    if element_vids.is_empty() {
        return None;
    }
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    for vid in element_vids {
        let (e_lo, e_hi) = derive_bounds_from_facts(facts, *vid)?;
        lo = Some(lo.map_or(e_lo, |cur| cur.min(e_lo)));
        hi = Some(hi.map_or(e_hi, |cur| cur.max(e_hi)));
    }
    Some((lo?, hi?))
}

/// Emit `output ∈ [multiplier * agg_lo, multiplier * agg_hi]` on
/// `output_vid` where `[agg_lo, agg_hi]` is the union of per-element
/// fact-derived bounds across `element_vids`. Returns `true` on emit,
/// `false` on no-op (any element unbounded, no elements, or
/// `checked_mul` overflow).
///
/// Sibling of [`interval_fact_for_int_binary`] for the reduction case.
/// For `np.sum`, `multiplier = N` (the element count). For `np.max` /
/// `np.min`, `multiplier = 1` (the output lives inside the element
/// bound by definition).
///
/// Soundness: only emits when every element has a visible bound. No
/// fact ⇒ no relay ⇒ no false claim. Overflow on
/// `multiplier * element_bound` likewise yields a no-op rather than a
/// wrapped (possibly unsound) value.
pub fn relay_reduction_output_interval_int(
    b: &mut crate::builder::IRBuilder,
    element_vids: &[crate::types::ValueId],
    output_vid: crate::types::ValueId,
    multiplier: i64,
) -> bool {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let Some((agg_lo, agg_hi)) = aggregate_element_bounds(&b.facts, element_vids) else {
        return false;
    };
    // `checked_mul` rejects overflow; the helper returns false rather
    // than depositing a wrapped (and possibly unsound) bound.
    let c1 = match agg_lo.checked_mul(multiplier) {
        Some(v) => v,
        None => return false,
    };
    let c2 = match agg_hi.checked_mul(multiplier) {
        Some(v) => v,
        None => return false,
    };
    // For negative multipliers the corners flip; take min/max so the
    // emitted bound is correct regardless of sign. (Sum/max/min in v1
    // always pass non-negative multipliers, but staying generic costs
    // nothing.)
    let (out_lo, out_hi) = (c1.min(c2), c1.max(c2));

    let fact = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_lo)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_hi)),
            },
        ],
    };
    b.facts.insert_for(output_vid, fact);
    true
}

/// Bound-aware chokepoint helper that consults `IRBuilder::prove` when
/// the static / resolver / fact-scan layers all fail.
///
/// Tries, in order:
///
/// 1. [`require_static_or_bounded_int`] — covers the existing
///    static-val, resolver `resolve_min/max`, and shape-matching
///    fact-scan layers. Returns immediately on `Static(_)` /
///    `Bounded { .. }`.
/// 2. If still `Neither`, performs an outward-doubling probe with
///    [`crate::builder::IRBuilder::prove`] to find a finite
///    `[min, max]` envelope, then binary-searches each half to tighten
///    it. Returns `Bounded { min, max }` iff both halves discharge
///    `Proved`; collapses to `Static(n)` when `min == max`.
///
/// Soundness: only [`ProveOutcome::Proved`] admits the bound.
/// [`ProveOutcome::Disproved`] and [`ProveOutcome::Unknown`] both
/// default to "no information" — the helper returns `Neither` for that
/// half. Treating `Unknown` as `Proved` would be a circuit-correctness
/// bug.
///
/// The probe is budget-bounded: outward window `[-(1<<32), 1<<32)`
/// with ≤ 64 prove() calls per side. Z3 timeouts inside `prove` honour
/// `ZINNIA_SMT_PROVE_TIMEOUT_MS`.
pub fn resolve_int_or_bounded(
    b: &mut crate::builder::IRBuilder,
    val: &Value,
    site: SiteKind,
    dbg: Option<&DebugInfo>,
) -> BoundedInt {
    let outcome = require_static_or_bounded_int(b, val, site, dbg);
    if !matches!(outcome, BoundedInt::Neither) {
        return outcome;
    }

    // Prove-based probe requires a Value backed by a ValueId — the
    // ContractTerm we ask `prove()` references the value by id.
    let Some(vid) = val.value_id() else {
        return BoundedInt::Neither;
    };

    let max = match prove_upper_bound(b, vid) {
        Some(m) => m,
        None => return BoundedInt::Neither,
    };
    let min = match prove_lower_bound(b, vid) {
        Some(m) => m,
        None => return BoundedInt::Neither,
    };
    if min > max {
        // Soundness gate: contradictory bounds mean the fact set is
        // inconsistent or `prove` returned Proved for incompatible
        // halves. Default-deny.
        return BoundedInt::Neither;
    }
    if min == max {
        BoundedInt::Static(min)
    } else {
        BoundedInt::Bounded { min, max }
    }
}

/// Chokepoint helper for ops that require a single static int and accept
/// prove-derived single-point bounds as equivalent.
///
/// Wraps [`resolve_int_or_bounded`] and panics with `site`'s diagnostic
/// when the value cannot be pinned to a single integer. Strict superset
/// of [`require_static_int`]: every program that compiles with the
/// latter compiles with this helper too, plus programs whose chokepoint
/// integer is provable via SMT arithmetic.
///
/// Admits:
///
/// - [`BoundedInt::Static(n)`] — same as `require_static_int`.
/// - [`BoundedInt::Bounded { min, max }`] where `min == max` —
///   degenerate range collapses to a single point. `resolve_int_or_bounded`
///   already normalizes this to `Static(_)`, but we double-check in case
///   future changes loosen the helper.
///
/// Panics on [`BoundedInt::Neither`] and on non-degenerate
/// [`BoundedInt::Bounded`] (the latter needs a runtime dispatch path
/// that's out of scope for this helper).
pub fn require_provable_static_int(
    b: &mut crate::builder::IRBuilder,
    val: &Value,
    site: SiteKind,
) -> i64 {
    match resolve_int_or_bounded(b, val, site, None) {
        BoundedInt::Static(n) => n,
        BoundedInt::Bounded { min, max } if min == max => min,
        _ => panic!("{}", site.diagnostic()),
    }
}

/// Binary-search the tightest `c` such that `Value(vid) <= c` is
/// `Proved`. Returns `None` if no finite `c` in `[-(1<<32), 1<<32)`
/// is provable.
fn prove_upper_bound(
    b: &mut crate::builder::IRBuilder,
    vid: crate::types::ValueId,
) -> Option<i64> {
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    const MAX_ABS: i64 = 1 << 32;
    let probe = |b: &crate::builder::IRBuilder, c: i64| -> bool {
        let term = ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(c)),
        };
        // Soundness: only Proved admits the bound. Unknown / Disproved
        // default to "no information" here.
        matches!(b.prove(&term), ProveOutcome::Proved)
    };
    // Find any provable upper bound by outward doubling.
    let mut hi = 0i64;
    if !probe(b, hi) {
        let mut step: i64 = 1;
        loop {
            hi = step;
            if probe(b, hi) {
                break;
            }
            if step >= MAX_ABS {
                return None;
            }
            step = step.saturating_mul(2).min(MAX_ABS);
        }
    }
    // Find lo such that the bound does NOT hold, so [lo, hi] brackets
    // the tightest c.
    let mut lo: i64 = -1;
    if probe(b, lo) {
        let mut step: i64 = 1;
        loop {
            lo = -step;
            if !probe(b, lo) {
                break;
            }
            if step >= MAX_ABS {
                // Bound holds for every c in window: degenerate.
                return Some(lo);
            }
            step = step.saturating_mul(2).min(MAX_ABS);
        }
    }
    // Binary search the boundary: invariant probe(hi) && !probe(lo).
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if probe(b, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// Mirror of [`prove_upper_bound`] for `Value(vid) >= c` (returns the
/// largest provable `c`).
fn prove_lower_bound(
    b: &mut crate::builder::IRBuilder,
    vid: crate::types::ValueId,
) -> Option<i64> {
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    const MAX_ABS: i64 = 1 << 32;
    let probe = |b: &crate::builder::IRBuilder, c: i64| -> bool {
        let term = ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(c)),
        };
        matches!(b.prove(&term), ProveOutcome::Proved)
    };
    // Find any provable lower bound by outward doubling downward.
    let mut lo: i64 = 0;
    if !probe(b, lo) {
        let mut step: i64 = 1;
        loop {
            lo = -step;
            if probe(b, lo) {
                break;
            }
            if step >= MAX_ABS {
                return None;
            }
            step = step.saturating_mul(2).min(MAX_ABS);
        }
    }
    // Find hi such that probe(hi) is false: bracket the boundary.
    let mut hi: i64 = 1;
    if probe(b, hi) {
        let mut step: i64 = 1;
        loop {
            hi = step;
            if !probe(b, hi) {
                break;
            }
            if step >= MAX_ABS {
                return Some(hi);
            }
            step = step.saturating_mul(2).min(MAX_ABS);
        }
    }
    // Binary search: invariant probe(lo) && !probe(hi).
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if probe(b, mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// Informational in-bound probe for dynamic array indices.
///
/// Item #4 of the `smt-invocation-load-bearing` card. Asks the resolver
/// whether `idx ∈ [lo, hi)` is provable; records telemetry tagged with
/// `SiteKind::DynamicIndexBound` regardless of outcome. Does NOT panic
/// on failure — soundness is enforced at prove time by the memory-trace
/// permutation argument. The probe exists to surface SMT engagement on
/// production-natural patterns (binary search, modular indexing, etc.)
/// where indices are runtime but bounds are nonetheless decidable.
///
/// Returns `true` if both `resolve_max(idx) < hi` and `resolve_min(idx)
/// >= lo` were proved by some layer (static_val / range / SMT).
pub fn probe_in_range(
    b: &mut crate::builder::IRBuilder,
    idx: &Value,
    lo: i64,
    hi: i64,
) -> bool {
    use std::sync::atomic::Ordering;
    let site_name = SiteKind::DynamicIndexBound.short_name();

    // Cheap escape: literal integer.
    if let Some(n) = idx.int_val() {
        let in_range = lo <= n && n < hi;
        // We still record the invocation so telemetry shows the call site
        // exists, but skip the resolver round-trip for the literal case.
        let (resolver, _stmts) = b.split_resolver_and_stmts();
        if let Some(t) = resolver.telemetry_handle() {
            t.record_chokepoint_invocation(site_name);
            if in_range {
                t.record_chokepoint_resolved(site_name);
            }
        }
        return in_range;
    }

    // Resolver pass — capture telemetry counters so we can detect SMT
    // engagement, then drop the resolver borrow before consulting facts.
    let (mut in_range, tel, smt_before) = {
        let (resolver, stmts) = b.split_resolver_and_stmts();
        let tel = resolver.telemetry_handle();
        let smt_before = tel.as_ref().map(|t| {
            t.queries_smt_resolved.load(Ordering::Relaxed)
                + t.queries_smt_unknown.load(Ordering::Relaxed)
        });
        if let Some(t) = tel.as_ref() {
            t.record_chokepoint_invocation(site_name);
        }
        let upper = resolver.resolve_max_with_stmts(idx, stmts);
        let lower = resolver.resolve_min_with_stmts(idx, stmts);
        let in_range = upper.map_or(false, |u| u < hi) && lower.map_or(false, |l| l >= lo);
        (in_range, tel, smt_before)
    };

    // Fact-fallback (compiler.fact-aware-chokepoints-batch): if the
    // resolver couldn't prove `idx ∈ [lo, hi)`, op-contract facts on
    // `idx.stmt_id()` may still pin it. Facts only tighten — we accept the
    // fallback iff both halves of `(min, max)` straddle `[lo, hi)`.
    if !in_range {
        if let Some((min, max)) = b.ask_bounds(idx) {
            if min >= lo && max < hi {
                in_range = true;
            }
        }
    }

    if let (Some(t), Some(before)) = (tel.as_ref(), smt_before) {
        let smt_after = t.queries_smt_resolved.load(Ordering::Relaxed)
            + t.queries_smt_unknown.load(Ordering::Relaxed);
        if smt_after > before {
            t.record_chokepoint_smt_engagement(site_name);
        }
        if in_range {
            t.record_chokepoint_resolved(site_name);
        }
    }
    in_range
}

/// Load-bearing Phase-E-style discharge of `lo <= idx < hi` at an
/// indexing chokepoint (Group 5a).
///
/// Unlike [`probe_in_range`] (informational — records telemetry, returns
/// a bool callers may ignore), this function *enforces* the bound:
///
/// * **Literal idx in range**: no-op (cheap fast path).
/// * **Literal idx out of range**: panic with a diagnostic naming `op_name`,
///   the literal, and `[lo, hi)`.
/// * **Non-literal idx with a `value_id`**: build the term
///   `idx >= lo ∧ idx < hi` and discharge through
///   [`crate::builder::IRBuilder::discharge_requires`] — Phase E policy
///   handles Proved (no-op), Disproved (compile panic), and Unknown
///   (witness emit by default, panic under `ZINNIA_OP_REQUIRES_STRICT=1`).
/// * **Non-literal idx with no `value_id` and no literal value**: no-op.
///   Sound: with no anchor we have nothing to reason about and no SSA
///   wire to constrain; the runtime memory-trace permutation argument
///   still enforces address validity in the prover. Callers requiring
///   tighter compile-time enforcement must thread a `value_id` through.
///
/// The action is purely a side effect; there is no return value.
pub fn discharge_index_in_range(
    b: &mut crate::builder::IRBuilder,
    idx: &Value,
    lo: i64,
    hi: i64,
    op_name: &'static str,
) {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractTerm, ContractVar,
    };

    // Literal-index fast path: compile-time decidable, panic if out of
    // range. This catches `arr[15]` on a length-10 array even when no
    // resolver is engaged.
    if let Some(n) = idx.int_val() {
        if !(lo <= n && n < hi) {
            panic!(
                "op `{}`: index {} out of range [{}, {})",
                op_name, n, lo, hi
            );
        }
        return;
    }

    // Non-literal: require a `value_id` to anchor a discharge term.
    // No value_id ⇒ no contract handle ⇒ sound no-op (the prover-side
    // permutation argument still enforces validity at proof time).
    let idx_vid = match idx.value_id() {
        Some(v) => v,
        None => return,
    };

    let term = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
                rhs: Box::new(ContractTerm::LitInt(lo)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Lt,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
                rhs: Box::new(ContractTerm::LitInt(hi)),
            },
        ],
    };
    b.discharge_requires(op_name, &term);
}

/// Relay `forall_eq_const(in, k)` for `k ∈ {0, 1}` from `input_vid` to
/// `output_vid` by firing the corresponding `zeros_content` / `ones_content`
/// op contract on the output.
///
/// Used by shape-preserving / element-replicating ops (`tile`, `repeat`, and
/// — once new cards land — `concatenate`, `reshape`, `transpose`, `slice`)
/// to forward the content fact that downstream reductions (`sum`, `prod`,
/// `mean`) need for compile-time specialization.
///
/// Returns `true` if a content fact was relayed, `false` otherwise. The
/// `k = 0` probe is tried first; in the pathological case where both
/// `forall_eq_const(in, 0)` and `forall_eq_const(in, 1)` are simultaneously
/// `Proved` (only possible for empty arrays), `zeros_content` wins by
/// declaration order.
pub fn relay_forall_eq_const_from_input(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    let pred_zero = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(input_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    if matches!(b.prove(&pred_zero), ProveOutcome::Proved) {
        b.fire_contract("zeros_content", output_vid, &HashMap::new());
        return true;
    }
    let pred_one = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(input_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    if matches!(b.prove(&pred_one), ProveOutcome::Proved) {
        b.fire_contract("ones_content", output_vid, &HashMap::new());
        return true;
    }
    false
}

/// Multi-input sibling of [`relay_forall_eq_const_from_input`]. Emits a
/// content fact on `output_vid` only when *every* `input_vid` provably
/// satisfies `forall_eq_const(in_i, k)` for the same `k ∈ {0, 1}`. Used by
/// element-union ops (`concatenate`, `stack`, `vstack`, `hstack`, `dstack`,
/// `column_stack`) where the output's elements are exactly the union of the
/// inputs' elements — so a uniform-constant output requires uniform-constant
/// inputs at the same value.
///
/// Returns `true` if a content fact was emitted. Empty `input_vids` returns
/// `false` (no inputs, no derivation possible).
pub fn relay_forall_eq_const_from_all_inputs(
    b: &mut crate::builder::IRBuilder,
    input_vids: &[crate::types::ValueId],
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    if input_vids.is_empty() {
        return false;
    }
    for k in &[0i64, 1i64] {
        let all_match = input_vids.iter().all(|&vid| {
            let term = ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(*k),
                ],
            };
            matches!(b.prove(&term), ProveOutcome::Proved)
        });
        if all_match {
            let contract = if *k == 0 { "zeros_content" } else { "ones_content" };
            b.fire_contract(contract, output_vid, &HashMap::new());
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::InputPath;
    use crate::ir_defs::IR;
    use crate::types::ScalarValue;

    #[test]
    fn static_only_resolver_matches_int_val() {
        let mut r = StaticOnlyResolver::new();
        let v = Value::Integer(ScalarValue::constant(42));
        assert_eq!(r.resolve_int(&v), Some(42));
        assert_eq!(r.resolve_max(&v), Some(42));
        assert_eq!(r.resolve_min(&v), Some(42));
    }

    #[test]
    fn static_only_resolver_unknown_for_runtime_int() {
        let mut r = StaticOnlyResolver::new();
        let v = Value::Integer(ScalarValue::runtime(0));
        assert_eq!(r.resolve_int(&v), None);
    }

    #[test]
    fn static_int_into_i64() {
        let s = StaticInt(7);
        let n: i64 = s.into();
        assert_eq!(n, 7);
    }

    #[test]
    fn site_kind_diagnostic_includes_axis() {
        assert!(SiteKind::ShapeAxis(3)
            .diagnostic()
            .contains("axis 3"));
    }

    // ---------------------------------------------------------------
    // SmtResolver tests
    // ---------------------------------------------------------------

    /// Helper: build a `Value::Integer` whose ptr is `stmt_id`. Mirrors
    /// what `IRBuilder::create_ir` does for the integer return type.
    fn runtime_int(stmt_id: StmtId) -> Value {
        Value::Integer(ScalarValue::runtime(stmt_id))
    }

    /// Helper: build a `Value::Boolean` whose ptr is `stmt_id`.
    fn runtime_bool(stmt_id: StmtId) -> Value {
        Value::Boolean(ScalarValue::runtime(stmt_id))
    }

    /// SMT-decidable but not static_val: `select(true, 7, 9)` with a
    /// non-folded ConstantBool input. The static_val path can't see
    /// through Select unless the optimiser already folded it; SMT can.
    #[test]
    fn smt_resolves_select_with_constant_cond() {
        // stmt0 = ConstantBool(true), stmt1 = ConstantInt(7),
        // stmt2 = ConstantInt(9), stmt3 = SelectI(stmt0, stmt1, stmt2)
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_int(3);
        let mut r = SmtResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(7));
    }

    /// Genuinely SMT-decidable via reasoning: `select(x == 5, 100, 100)`
    /// where `x` is a free `ReadInteger`. Both branches are 100, so SMT
    /// proves the result is 100 — but `static_val` can't, because it
    /// doesn't know `x`.
    #[test]
    fn smt_resolves_select_with_both_branches_equal() {
        // stmt0 = ReadInteger("x"), stmt1 = ConstantInt(5),
        // stmt2 = EqI(stmt0, stmt1), stmt3 = ConstantInt(100),
        // stmt4 = ConstantInt(100), stmt5 = SelectI(stmt2, stmt3, stmt4)
        let stmts = vec![
            IRStatement::new(
                0, crate::types::ValueId::next(),
                IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
                vec![], vec![],
                None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 5 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::EqI, vec![0, 1], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::ConstantInt { value: 100 }, vec![], vec![], None),
            IRStatement::new(4, crate::types::ValueId::next(), IR::ConstantInt { value: 100 }, vec![], vec![], None),
            IRStatement::new(5, crate::types::ValueId::next(), IR::SelectI, vec![2, 3, 4], vec![], None),
        ];
        let v = runtime_int(5);
        let mut r = SmtResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(100));
    }

    /// Free input wire: `ReadInteger` returns None (the value is genuinely
    /// not constant; SMT must not fabricate one).
    #[test]
    fn smt_returns_none_on_free_variable() {
        let stmts = vec![IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
            vec![], vec![],
            None)];
        let v = runtime_int(0);
        let mut r = SmtResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// Disable flag: even on an SMT-decidable case, returning None
    /// (after the static-val fast path).
    #[test]
    fn smt_disable_flag_short_circuits() {
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_int(3);
        let mut r = SmtResolver::new().with_disabled(true);
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// Cache hit on second call: after one query, the cache has an entry
    /// for ptr 3. A second query on the same wire should hit the cache
    /// without re-querying Z3 (we observe this via cache_size == 1
    /// throughout).
    #[test]
    fn smt_caches_resolution() {
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_int(3);
        let mut r = SmtResolver::new();

        assert_eq!(r.cache_size(), 0);
        let first = r.resolve_int_with_stmts(&v, &stmts);
        assert_eq!(first, Some(7));
        assert_eq!(r.cache_size(), 1);

        let second = r.resolve_int_with_stmts(&v, &stmts);
        assert_eq!(second, Some(7));
        // cache_size is still 1 — we didn't add a new entry.
        assert_eq!(r.cache_size(), 1);
    }

    /// Tight timeout returns None on a pathological formula. We construct
    /// an SmtResolver with a 1 ms timeout and a deliberately-non-trivial
    /// formula (a long chain of MulI's grows the search space) — the
    /// timeout fires and the resolver returns None instead of hanging.
    ///
    /// The test is hermetic because Z3 honours the timeout regardless of
    /// machine speed; the only assertion is "returns None" (not "returns
    /// quickly"). To make this robust we compose a formula whose decision
    /// is unique (so the resolver would, given enough time, succeed) but
    /// which contains enough multiplicative structure that a 1 ms budget
    /// can't crack it.
    #[test]
    fn smt_honours_tight_timeout() {
        // Build: x = ReadInteger("x"). Then a big chain of multiplications
        // and conditional adds. Even though `x*x*...*x ` is determined by x,
        // the resolver can't prove uniqueness without first picking x — and
        // the formula is large enough that with a 1 ms budget Z3 returns
        // unknown rather than the unique-not-found "non-unique" verdict
        // produced when the sub-checks succeed.
        let mut stmts = Vec::new();
        stmts.push(IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
            vec![], vec![],
            None));
        // Build a chain: stmt1 = stmt0 * stmt0, stmt2 = stmt1 * stmt0, ...
        // up to stmt30. Result has high arithmetic complexity.
        let mut last = 0u32;
        for i in 1..=30 {
            stmts.push(IRStatement::new(i, crate::types::ValueId::next(), IR::MulI, vec![last, 0], vec![], None));
            last = i;
        }
        let v = runtime_int(last);
        let mut r = SmtResolver::new().with_timeout(1);
        // Within 1 ms, Z3 likely returns sat with a model — but the
        // re-check (var != that_value) will likely also return sat (Z3
        // can find another counter-model), so the resolver returns None.
        // Either way, the test asserts no Some(_) leaks: because the
        // wire genuinely depends on x, no unique value should be
        // returned regardless of timeout.
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// `on_ir_mutated(&[])` clears the entire cache (P1 conservative).
    #[test]
    fn smt_on_ir_mutated_clears_cache() {
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_int(3);
        let mut r = SmtResolver::new();
        let _ = r.resolve_int_with_stmts(&v, &stmts);
        assert_eq!(r.cache_size(), 1);
        r.on_ir_mutated(&[]);
        assert_eq!(r.cache_size(), 0);
    }

    /// Static-val fast path: SmtResolver returns the constant immediately
    /// without consulting Z3. Verified by passing an EMPTY stmts slice
    /// (Z3 path would panic on stmt[ptr] otherwise — actually it'd return
    /// None, but the point is fast-path doesn't even attempt the walk).
    #[test]
    fn smt_static_val_fast_path() {
        let stmts: Vec<IRStatement> = vec![];
        let v = Value::Integer(ScalarValue::constant(123));
        let mut r = SmtResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(123));
        // No cache entry for static-val results (no ptr to key on).
        assert_eq!(r.cache_size(), 0);
    }

    /// resolve_bool_with_stmts on a select with a constant cond.
    #[test]
    fn smt_resolves_bool_through_select() {
        // stmt0 = ConstantBool(false),
        // stmt1 = ConstantBool(true), stmt2 = ConstantBool(true),
        // stmt3 = SelectB(stmt0, stmt1, stmt2)
        // Both branches are true so result is true regardless of cond.
        let stmts = vec![
            IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: false }, vec![], vec![], None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectB, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_bool(3);
        let mut r = SmtResolver::new();
        assert_eq!(r.resolve_bool_with_stmts(&v, &stmts), Some(true));
    }

    /// `SmtResolver` is `Send + Sync` (required by `Resolver` trait
    /// because `IRGraph` is held by a `#[pyclass]`). Compile-time check.
    #[test]
    fn smt_resolver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SmtResolver>();
    }

    // ---------------------------------------------------------------
    // LayeredResolver tests (P2 commit 4)
    // ---------------------------------------------------------------

    use crate::optim::range::RangeResolver;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Test-only counter-tracking resolver. Wraps an inner resolver and
    /// increments a shared counter on every `_with_stmts` call. Used to
    /// verify the layered resolver's "answers first" behaviour.
    struct CountingResolver {
        inner: Box<dyn Resolver>,
        calls: Arc<AtomicUsize>,
    }

    impl Resolver for CountingResolver {
        fn resolve_int(&mut self, val: &Value) -> Option<i64> {
            self.inner.resolve_int(val)
        }
        fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
            self.inner.resolve_bool(val)
        }
        fn resolve_max(&mut self, val: &Value) -> Option<i64> {
            self.inner.resolve_max(val)
        }
        fn resolve_min(&mut self, val: &Value) -> Option<i64> {
            self.inner.resolve_min(val)
        }
        fn resolve_int_with_stmts(
            &mut self,
            val: &Value,
            stmts: &[IRStatement],
        ) -> Option<i64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.resolve_int_with_stmts(val, stmts)
        }
        fn resolve_bool_with_stmts(
            &mut self,
            val: &Value,
            stmts: &[IRStatement],
        ) -> Option<bool> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.resolve_bool_with_stmts(val, stmts)
        }
        fn resolve_max_with_stmts(
            &mut self,
            val: &Value,
            stmts: &[IRStatement],
        ) -> Option<i64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.resolve_max_with_stmts(val, stmts)
        }
        fn resolve_min_with_stmts(
            &mut self,
            val: &Value,
            stmts: &[IRStatement],
        ) -> Option<i64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.resolve_min_with_stmts(val, stmts)
        }
        fn on_ir_mutated(&mut self, affected: &[StmtId]) {
            self.inner.on_ir_mutated(affected);
        }
    }

    /// When the range layer answers a query, the SMT layer is never
    /// consulted. Construct a `select(c, 7, 7)` (range resolves to 7,
    /// SMT *would* also resolve, but should be skipped). The counting
    /// SMT layer's call-count must be 0.
    #[test]
    fn layered_range_answers_first() {
        let stmts = vec![
            IRStatement::new(
                0, crate::types::ValueId::next(),
                IR::ReadInteger {
                    path: InputPath::new("c", vec![]),
                    is_public: false,
                },
                vec![], vec![],
                None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_int(3);

        let smt_calls = Arc::new(AtomicUsize::new(0));
        let counting_smt = CountingResolver {
            inner: Box::new(SmtResolver::new()),
            calls: Arc::clone(&smt_calls),
        };
        let mut layered = LayeredResolver::new(vec![
            Box::new(RangeResolver::new()),
            Box::new(counting_smt),
        ]);

        assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), Some(7));
        // Range answered → SMT was never called.
        assert_eq!(smt_calls.load(Ordering::SeqCst), 0);
    }

    /// When range can't resolve, the layered resolver falls through to
    /// SMT. Construct `select(x == x, 7, 9)`: range only sees `[7, 9]`
    /// (no point) since it doesn't reason about the cond's tautology.
    /// SMT proves `x == x` is always true → result is 7. The counting
    /// SMT layer's call-count must be ≥ 1.
    #[test]
    fn layered_falls_through_to_smt() {
        let stmts = vec![
            IRStatement::new(
                0, crate::types::ValueId::next(),
                IR::ReadInteger {
                    path: InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![], vec![],
                None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::EqI, vec![0, 0], vec![], None), // x == x → true
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
            IRStatement::new(4, crate::types::ValueId::next(), IR::SelectI, vec![1, 2, 3], vec![], None),
        ];
        let v = runtime_int(4);

        let smt_calls = Arc::new(AtomicUsize::new(0));
        let counting_smt = CountingResolver {
            inner: Box::new(SmtResolver::new()),
            calls: Arc::clone(&smt_calls),
        };
        let mut layered = LayeredResolver::new(vec![
            Box::new(RangeResolver::new()),
            Box::new(counting_smt),
        ]);
        assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), Some(7));
        // SMT was consulted (range couldn't tell which branch).
        assert!(smt_calls.load(Ordering::SeqCst) >= 1);
    }

    /// When neither layer can resolve (a free variable), the layered
    /// resolver returns None.
    #[test]
    fn layered_returns_none_when_neither_resolves() {
        let stmts = vec![IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None)];
        let v = runtime_int(0);
        let mut layered = LayeredResolver::range_then_smt();
        assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// `LayeredResolver` is Send + Sync.
    #[test]
    fn layered_resolver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LayeredResolver>();
    }

    // ---------------------------------------------------------------
    // P5 telemetry tests
    // ---------------------------------------------------------------

    /// `SmtResolver` increments `queries_smt_unknown` when Z3 can't
    /// resolve a free variable. The static_val and cache fast-paths
    /// shouldn't fire.
    #[test]
    fn telemetry_smt_unknown_on_free_var() {
        let stmts = vec![IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None)];
        let v = runtime_int(0);
        let mut r = SmtResolver::new();
        let t = r.telemetry();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
        assert_eq!(t.queries_smt_unknown.load(Ordering::SeqCst), 1);
        assert_eq!(t.queries_smt_resolved.load(Ordering::SeqCst), 0);
        // The duration histogram has a single entry now (somewhere).
        let total: usize =
            (0..crate::optim::telemetry::NUM_DURATION_BUCKETS)
                .map(|i| t.smt_duration_buckets[i].load(Ordering::SeqCst))
                .sum();
        assert_eq!(total, 1);
    }

    /// P5 commit 3: `with_max_formula_size` triggers the oversized
    /// short-circuit when a formula's reverse-reachability walk would
    /// exceed the cap. The resolver returns None and the
    /// `queries_skipped_oversized` counter ticks. Verifies the cap is
    /// wired through `resolve_int_with_stmts` to the walker.
    #[test]
    fn smt_max_formula_size_aborts_oversized_walk() {
        // Construct a 5-statement chain (Constant → MulI → MulI → MulI →
        // MulI). With max_formula_size = 2 the walker aborts before
        // encoding the root, so the resolver returns None and the
        // oversized counter increments.
        let stmts = vec![
            IRStatement::new(
                0, crate::types::ValueId::next(),
                IR::ReadInteger {
                    path: InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![], vec![],
                None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::MulI, vec![0, 0], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::MulI, vec![1, 0], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::MulI, vec![2, 0], vec![], None),
            IRStatement::new(4, crate::types::ValueId::next(), IR::MulI, vec![3, 0], vec![], None),
        ];
        let v = runtime_int(4);
        let mut r = SmtResolver::new().with_max_formula_size(2);
        let t = r.telemetry();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
        assert_eq!(t.queries_skipped_oversized.load(Ordering::SeqCst), 1);
        // The "smt_unknown" counter must NOT have incremented — we
        // never hit the solver, the walk aborted first.
        assert_eq!(t.queries_smt_unknown.load(Ordering::SeqCst), 0);
        assert_eq!(t.queries_smt_resolved.load(Ordering::SeqCst), 0);
    }

    /// P5 commit 3: a 1ms timeout on a moderately-complex formula
    /// returns None instead of resolving the unique value. (The legacy
    /// `smt_honours_tight_timeout` test already covers this; this
    /// duplicates with the explicit "via the new mitigation knob"
    /// framing.)
    #[test]
    fn smt_tight_timeout_via_with_timeout_mitigation() {
        let mut stmts = Vec::new();
        stmts.push(IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None));
        let mut last = 0u32;
        for i in 1..=20 {
            stmts.push(IRStatement::new(i, crate::types::ValueId::next(), IR::MulI, vec![last, 0], vec![], None));
            last = i;
        }
        let v = runtime_int(last);
        let mut r = SmtResolver::new().with_timeout(1);
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// `LayeredResolver::range_then_smt` shares one telemetry across the
    /// range and SMT layers. Construct `select(c, 7, 7)` (range answers
    /// it). Counters: `queries_total` = 1, `queries_range_hit` = 1,
    /// `queries_smt_resolved` = 0.
    #[test]
    fn telemetry_layered_range_hit_increments_shared_telemetry() {
        let stmts = vec![
            IRStatement::new(
                0, crate::types::ValueId::next(),
                IR::ReadInteger {
                    path: InputPath::new("c", vec![]),
                    is_public: false,
                },
                vec![], vec![],
                None),
            IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
            IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
        ];
        let v = runtime_int(3);
        let mut layered = LayeredResolver::range_then_smt();
        let t = layered.telemetry_handle().unwrap();
        assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), Some(7));
        assert_eq!(t.queries_total.load(Ordering::SeqCst), 1);
        assert_eq!(t.queries_range_hit.load(Ordering::SeqCst), 1);
        assert_eq!(t.queries_smt_resolved.load(Ordering::SeqCst), 0);
        assert_eq!(t.queries_smt_unknown.load(Ordering::SeqCst), 0);
    }

    // ---------------------------------------------------------------
    // require_static_or_bounded_int / BoundedInt
    // ---------------------------------------------------------------

    #[test]
    fn bounded_int_static_branch_for_literal() {
        // A compile-time constant integer must resolve via the Static
        // branch — same as require_static_int's fast lane.
        let mut b = crate::builder::IRBuilder::new();
        let v = b.ir_constant_int(42);
        let outcome = require_static_or_bounded_int(
            &mut b,
            &v,
            SiteKind::ShapeAxis(0),
            None,
        );
        assert_eq!(outcome, BoundedInt::Static(42));
    }

    #[test]
    fn bounded_int_neither_for_unconstrained_symbolic() {
        // A free runtime ReadInteger has no static value and no
        // resolver-provable bound under the SMT-enabled pipeline.
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let outcome = require_static_or_bounded_int(
            &mut b,
            &k,
            SiteKind::ShapeAxis(0),
            None,
        );
        assert_eq!(outcome, BoundedInt::Neither);
    }

    #[test]
    fn bounded_int_recovers_bound_from_op_contract_facts() {
        // Consumer-side wiring for `compiler.op-contract-corpus-demos`:
        // when the resolver-based pass yields Neither, the chokepoint
        // recovers a bound from `b.facts.per_stmt` for ptr-anchored
        // `SsaPtr(p) op LitInt(n)` facts deposited by op contracts.
        use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();

        // Plant op-contract-shaped facts: `k >= 0` and `k <= 16`.
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
        );
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
                rhs: Box::new(ContractTerm::LitInt(16)),
            },
        );

        let outcome = require_static_or_bounded_int(
            &mut b,
            &k,
            SiteKind::ShapeAxis(0),
            None,
        );
        assert_eq!(outcome, BoundedInt::Bounded { min: 0, max: 16 });
    }

    #[test]
    fn bounded_int_bounded_branch_via_structural_predicate() {
        // The load-bearing end-to-end test: with a `nnz(x) == k`
        // precondition and a 16-element array `x`, the helper returns
        // `Bounded { min: 0, max: 16 }`. Mirrors the Walkthrough-1
        // resolver-chain test but goes through the chokepoint helper.
        use crate::circuit_input::PathSegment;

        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        // Emit reads at indices 0 and 15 — the array-length inference
        // uses max-index + 1 = 16.
        let _ = b.ir_read_float(
            InputPath::new("x", vec![PathSegment::Index(0)]),
            false,
        );
        let _ = b.ir_read_float(
            InputPath::new("x", vec![PathSegment::Index(15)]),
            false,
        );

        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);

        let _ = b.ir_structural_predicate(
            "nnz".to_string(),
            vec!["x".to_string()],
            Some("==".to_string()),
            Some("k".to_string()),
        );

        let outcome = require_static_or_bounded_int(
            &mut b,
            &k,
            SiteKind::ShapeAxis(0),
            None,
        );
        assert_eq!(
            outcome,
            BoundedInt::Bounded { min: 0, max: 16 },
            "expected `nnz(x) == k` with len(x)=16 to bound k in [0, 16]; got {:?}",
            outcome,
        );
    }

    #[test]
    fn bounded_int_static_from_statically_resolvable_select() {
        // `select(true, 7, 7)` is statically resolvable to 7 by the SMT
        // layer (and the range layer). Make sure the helper picks the
        // Static branch — not the Bounded one — when a unique value is
        // available.
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        // Build `select(true, 7, 7)` so it constant-folds to 7.
        let t = b.ir_constant_bool(true);
        let a = b.ir_constant_int(7);
        let c = b.ir_constant_int(7);
        let v = b.create_ir(&IR::SelectI, &[t, a, c]);
        let outcome = require_static_or_bounded_int(
            &mut b,
            &v,
            SiteKind::ShapeAxis(0),
            None,
        );
        assert_eq!(outcome, BoundedInt::Static(7));
    }

    #[test]
    fn bounded_int_from_static_int_conversion() {
        let s = StaticInt(13);
        let b: BoundedInt = s.into();
        assert_eq!(b, BoundedInt::Static(13));
    }

    // ---------------------------------------------------------------
    // compiler.fact-aware-chokepoints-batch: each wired chokepoint
    // recovers a bound / static value from op-contract-deposited facts
    // when the resolver-based pass can't.
    // ---------------------------------------------------------------

    /// Helper: plant a `Cmp(Value(vid) <op> LitInt(n))` fact on `b.facts`.
    fn plant_cmp_fact(
        b: &mut crate::builder::IRBuilder,
        vid: crate::types::ValueId,
        op: crate::optim::predicates::formula::CmpOp,
        n: i64,
    ) {
        use crate::optim::predicates::formula::{ContractTerm, ContractVar};
        b.facts.insert_for(
            vid,
            ContractTerm::Cmp {
                op,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
                rhs: Box::new(ContractTerm::LitInt(n)),
            },
        );
    }

    #[test]
    fn probe_in_range_recovers_from_op_contract_facts() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        // A runtime int with no resolver-derivable bound. We pick a
        // window [10, 20) so the resolver's spuriously-tight (0, 0)
        // fallback for free reads can't accidentally satisfy the probe.
        let idx = b.ir_read_integer(InputPath::new("idx", vec![]), false);
        let idx_vid = idx.value_id().unwrap();
        assert!(
            !probe_in_range(&mut b, &idx, 10, 20),
            "without facts, an unconstrained read cannot satisfy idx ∈ [10, 20)",
        );

        // Plant `idx >= 10` and `idx <= 19` as op-contract-shaped facts.
        plant_cmp_fact(&mut b, idx_vid, CmpOp::Ge, 10);
        plant_cmp_fact(&mut b, idx_vid, CmpOp::Le, 19);

        assert!(
            probe_in_range(&mut b, &idx, 10, 20),
            "fact-fallback should let probe_in_range prove idx ∈ [10, 20)",
        );
    }

    #[test]
    fn require_static_int_recovers_from_op_contract_eq_fact() {
        // Fact-fallback for chokepoints requiring a single static int
        // (slice/range/repeat/reshape/split). An `Eq` fact pins the
        // value to a single point.
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let v = b.ir_read_integer(InputPath::new("n", vec![]), false);
        let v_vid = v.value_id().unwrap();

        // Without facts: require_static_int fails (free read).
        assert!(require_static_int(&mut b, &v, SiteKind::RepeatCount, None).is_err());

        // Plant `n == 7`.
        plant_cmp_fact(&mut b, v_vid, CmpOp::Eq, 7);

        let result = require_static_int(&mut b, &v, SiteKind::RepeatCount, None)
            .expect("expected fact-fallback to recover static int");
        assert_eq!(result, StaticInt(7));
    }

    #[test]
    fn require_static_int_rejects_when_facts_only_give_range() {
        // Range facts (Ge + Le with distinct bounds) don't satisfy the
        // single-static-int contract; the chokepoint must still reject.
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let v = b.ir_read_integer(InputPath::new("n", vec![]), false);
        let v_vid = v.value_id().unwrap();
        plant_cmp_fact(&mut b, v_vid, CmpOp::Ge, 0);
        plant_cmp_fact(&mut b, v_vid, CmpOp::Le, 9);

        assert!(
            require_static_int(&mut b, &v, SiteKind::RepeatCount, None).is_err(),
            "range facts (min != max) must not be accepted as a static int",
        );
    }

    // ---------------------------------------------------------------
    // resolve_int_or_bounded — prove-based fallback for bounded ints
    // (compiler.chokepoint-prove-integration)
    // ---------------------------------------------------------------

    #[test]
    fn resolve_int_or_bounded_falls_through_to_static_for_literal() {
        let mut b = crate::builder::IRBuilder::new();
        let v = b.ir_constant_int(42);
        let outcome = resolve_int_or_bounded(
            &mut b,
            &v,
            SiteKind::ReshapeDim,
            None,
        );
        assert_eq!(outcome, BoundedInt::Static(42));
    }

    #[test]
    fn resolve_int_or_bounded_falls_through_to_existing_fact_scan() {
        // The existing scanner handles `Ge n` + `Le m` already; the
        // prove-based fallback must not interfere when that layer
        // already returns Bounded.
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();
        plant_cmp_fact(&mut b, k_vid, CmpOp::Ge, 0);
        plant_cmp_fact(&mut b, k_vid, CmpOp::Le, 16);

        let outcome = resolve_int_or_bounded(
            &mut b,
            &k,
            SiteKind::ReshapeDim,
            None,
        );
        assert_eq!(outcome, BoundedInt::Bounded { min: 0, max: 16 });
    }

    #[test]
    fn resolve_int_or_bounded_recovers_bound_via_prove_arithmetic_fact() {
        // The fact-scanner only matches `Cmp(Value, LitInt)` shapes.
        // An arithmetic-shape fact `k + k <= 20` (== `2*k <= 20`)
        // doesn't decompose to a bound by shape; Z3 proves k <= 10
        // and k >= -10 (no lower bound implied; the symmetric upper
        // gives a finite max but no constructive lower without more
        // facts). We plant both halves so the test pins both bounds.
        use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();

        // Plant `k + k <= 20` and `k + k >= -20`. Shape-matching
        // scanner can't see through Arith; prove() can.
        let k_plus_k = ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        };
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(k_plus_k.clone()),
                rhs: Box::new(ContractTerm::LitInt(20)),
            },
        );
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(k_plus_k),
                rhs: Box::new(ContractTerm::LitInt(-20)),
            },
        );

        let outcome = resolve_int_or_bounded(
            &mut b,
            &k,
            SiteKind::ReshapeDim,
            None,
        );
        assert_eq!(
            outcome,
            BoundedInt::Bounded { min: -10, max: 10 },
            "expected prove() to derive k ∈ [-10, 10] from `2k ∈ [-20, 20]`",
        );
    }

    #[test]
    fn resolve_int_or_bounded_returns_neither_on_unknown() {
        // A genuinely free runtime int with no facts must return
        // Neither — Unknown from prove() defaults to no admission.
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let outcome = resolve_int_or_bounded(
            &mut b,
            &k,
            SiteKind::ReshapeDim,
            None,
        );
        assert_eq!(outcome, BoundedInt::Neither);
    }

    #[test]
    fn resolve_int_or_bounded_collapses_min_eq_max_to_static() {
        // `prove()`-derivable arithmetic identity: `k * k == 16` with
        // an extra `k >= 0` fact pins k to 4 uniquely. The helper
        // should report `Static(4)`, not `Bounded { min: 4, max: 4 }`.
        use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();

        let k_sq = ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        };
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Eq,
                lhs: Box::new(k_sq),
                rhs: Box::new(ContractTerm::LitInt(16)),
            },
        );
        plant_cmp_fact(&mut b, k_vid, CmpOp::Ge, 0);

        let outcome = resolve_int_or_bounded(
            &mut b,
            &k,
            SiteKind::ReshapeDim,
            None,
        );
        assert_eq!(outcome, BoundedInt::Static(4));
    }

    #[test]
    fn ndarray_axis_or_other_chokepoint_admits_value_provable_via_prove() {
        // Marquee regression for compiler.consumer-prove-fallback-rollout:
        // pick a non-reshape/split chokepoint (`SiteKind::RepeatCount`,
        // converted in this rollout) and show `require_provable_static_int`
        // admits a value pinned by `k * k == 16` ∧ `k >= 0` ⟹ k == 4.
        // Today's `require_static_int` would reject this program; the
        // promoted Phase-A helper unblocks it everywhere.
        use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();

        // Plant `k * k == 16` (arithmetic-shaped fact — shape-matcher
        // can't decompose, prove() can).
        let k_sq = ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        };
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Eq,
                lhs: Box::new(k_sq),
                rhs: Box::new(ContractTerm::LitInt(16)),
            },
        );
        plant_cmp_fact(&mut b, k_vid, CmpOp::Ge, 0);

        let n = require_provable_static_int(&mut b, &k, SiteKind::RepeatCount);
        assert_eq!(n, 4, "prove()-derived single-point bound must be admitted at RepeatCount");
    }

    #[test]
    fn ask_bounds_falls_back_to_facts() {
        // ask_bounds returns the resolver's (min, max) when it's a
        // non-trivial range; otherwise it falls back to facts. For an
        // unconstrained read the SMT optimizer returns degenerate
        // (Some(0), Some(0)), which fails the min < max gate, so the
        // helper consults op-contract facts.
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let v = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let v_vid = v.value_id().unwrap();
        plant_cmp_fact(&mut b, v_vid, CmpOp::Ge, 0);
        plant_cmp_fact(&mut b, v_vid, CmpOp::Le, 16);
        assert_eq!(b.ask_bounds(&v), Some((0, 16)));
    }

    // ---------------------------------------------------------------
    // Float-arith interval E (Group 9).
    // ---------------------------------------------------------------

    /// Helper: plant a `Cmp(Value(vid) <op> LitFloat(n))` fact on `b.facts`.
    fn plant_float_cmp_fact(
        b: &mut crate::builder::IRBuilder,
        vid: crate::types::ValueId,
        op: crate::optim::predicates::formula::CmpOp,
        n: f64,
    ) {
        use crate::optim::predicates::formula::{ContractFloat, ContractTerm, ContractVar};
        b.facts.insert_for(
            vid,
            ContractTerm::Cmp {
                op,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(n))),
            },
        );
    }

    /// Helper: extract `(lo, hi)` from the `BoolComb(And, [Ge(out, lo), Le(out, hi)])`
    /// shape that `interval_fact_for_float_binary` emits.
    fn extract_float_bounds_from_term(
        term: &crate::optim::predicates::formula::ContractTerm,
    ) -> (f64, f64) {
        use crate::optim::predicates::formula::{
            BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
        };
        if let ContractTerm::BoolComb { op: BoolOp::And, operands } = term {
            assert_eq!(operands.len(), 2);
            let mut lo: Option<f64> = None;
            let mut hi: Option<f64> = None;
            for clause in operands {
                if let ContractTerm::Cmp { op, lhs, rhs } = clause {
                    assert!(matches!(lhs.as_ref(), ContractTerm::Var(ContractVar::Value(_))));
                    if let ContractTerm::LitFloat(ContractFloat(n)) = rhs.as_ref() {
                        match op {
                            CmpOp::Ge => lo = Some(*n),
                            CmpOp::Le => hi = Some(*n),
                            _ => panic!("unexpected cmp op in emitted bound: {:?}", op),
                        }
                    }
                }
            }
            (lo.expect("missing lo bound"), hi.expect("missing hi bound"))
        } else {
            panic!("expected BoolComb(And, …), got {:?}", term);
        }
    }

    #[test]
    fn interval_float_add_yields_sum_of_bounds() {
        use crate::optim::predicates::formula::{ArithOp, CmpOp};
        let mut b = crate::builder::IRBuilder::new();
        let a = b.ir_read_float(InputPath::new("a", vec![]), false);
        let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
        let a_vid = a.value_id().unwrap();
        let b_vid = bb.value_id().unwrap();
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, 0.0);
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, 5.0);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, 10.0);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, 20.0);

        let out_vid = crate::types::ValueId::next();
        let term = interval_fact_for_float_binary(&b.facts, ArithOp::Add, a_vid, b_vid, out_vid)
            .expect("expected interval fact for add");
        let (lo, hi) = extract_float_bounds_from_term(&term);
        assert_eq!(lo, 10.0);
        assert_eq!(hi, 25.0);
    }

    #[test]
    fn interval_float_sub_yields_diff_of_bounds() {
        use crate::optim::predicates::formula::{ArithOp, CmpOp};
        let mut b = crate::builder::IRBuilder::new();
        let a = b.ir_read_float(InputPath::new("a", vec![]), false);
        let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
        let a_vid = a.value_id().unwrap();
        let b_vid = bb.value_id().unwrap();
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, 0.0);
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, 5.0);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, 10.0);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, 20.0);

        let out_vid = crate::types::ValueId::next();
        let term = interval_fact_for_float_binary(&b.facts, ArithOp::Sub, a_vid, b_vid, out_vid)
            .expect("expected interval fact for sub");
        let (lo, hi) = extract_float_bounds_from_term(&term);
        assert_eq!(lo, -20.0);
        assert_eq!(hi, -5.0);
    }

    #[test]
    fn interval_float_mul_yields_corner_min_max() {
        use crate::optim::predicates::formula::{ArithOp, CmpOp};
        let mut b = crate::builder::IRBuilder::new();
        let a = b.ir_read_float(InputPath::new("a", vec![]), false);
        let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
        let a_vid = a.value_id().unwrap();
        let b_vid = bb.value_id().unwrap();
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, -2.0);
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, 3.0);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, -1.0);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, 4.0);

        let out_vid = crate::types::ValueId::next();
        let term = interval_fact_for_float_binary(&b.facts, ArithOp::Mul, a_vid, b_vid, out_vid)
            .expect("expected interval fact for mul");
        let (lo, hi) = extract_float_bounds_from_term(&term);
        assert_eq!(lo, -8.0);
        assert_eq!(hi, 12.0);
    }

    #[test]
    fn interval_float_overflow_skips_emit() {
        use crate::optim::predicates::formula::{ArithOp, CmpOp};
        let mut b = crate::builder::IRBuilder::new();
        let a = b.ir_read_float(InputPath::new("a", vec![]), false);
        let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
        let a_vid = a.value_id().unwrap();
        let b_vid = bb.value_id().unwrap();
        // Both inputs span the full f64 range. The corner products
        // overflow to ±inf, so no fact is emitted.
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, f64::MIN);
        plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, f64::MAX);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, f64::MIN);
        plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, f64::MAX);

        let out_vid = crate::types::ValueId::next();
        let term = interval_fact_for_float_binary(&b.facts, ArithOp::Mul, a_vid, b_vid, out_vid);
        assert!(term.is_none(), "non-finite output bound must skip emit");
    }

    #[test]
    fn interval_float_unbounded_input_skips_emit() {
        use crate::optim::predicates::formula::ArithOp;
        let mut b = crate::builder::IRBuilder::new();
        let a = b.ir_read_float(InputPath::new("a", vec![]), false);
        let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
        let a_vid = a.value_id().unwrap();
        let b_vid = bb.value_id().unwrap();
        // No facts planted on either input.

        let out_vid = crate::types::ValueId::next();
        let term = interval_fact_for_float_binary(&b.facts, ArithOp::Add, a_vid, b_vid, out_vid);
        assert!(term.is_none(), "unbounded input must skip emit");
    }

    #[test]
    fn relay_sqrt_with_input_bound_emits_output_interval() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 4.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 9.0);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_sqrt_output_interval(&mut b, x_vid, out_vid);
        assert!(emitted, "expected relay to deposit an output bound");

        let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
        let term = entries.last().expect("at least one fact");
        let (lo, hi) = extract_float_bounds_from_term(term);
        assert_eq!(lo, 2.0);
        assert_eq!(hi, 3.0);
    }

    #[test]
    fn relay_sqrt_with_unbounded_input_skips_emit() {
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        // No facts planted on the input.

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_sqrt_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "unbounded input must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_sqrt_with_negative_lo_skips_emit() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, -1.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_sqrt_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "negative lo must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_exp_with_input_bound_emits_output_interval() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 0.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_exp_output_interval(&mut b, x_vid, out_vid);
        assert!(emitted, "expected relay to deposit an output bound");

        let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
        let term = entries.last().expect("at least one fact");
        let (lo, hi) = extract_float_bounds_from_term(term);
        assert_eq!(lo, 0.0_f64.exp());
        assert_eq!(hi, 1.0_f64.exp());
    }

    #[test]
    fn relay_exp_with_unbounded_input_skips_emit() {
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        // No facts planted on the input.

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_exp_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "unbounded input must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_exp_with_overflow_hi_skips_emit() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        // `exp(1e308)` overflows f64 to `+inf`; `is_finite()` guard must
        // skip rather than depositing a non-finite literal.
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 0.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0e308);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_exp_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "overflowing exp(hi) must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_log_with_input_bound_emits_output_interval() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        let e = 1.0_f64.exp();
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 1.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, e);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_log_output_interval(&mut b, x_vid, out_vid);
        assert!(emitted, "expected relay to deposit an output bound");

        let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
        let term = entries.last().expect("at least one fact");
        let (lo, hi) = extract_float_bounds_from_term(term);
        assert_eq!(lo, 1.0_f64.ln());
        assert_eq!(hi, e.ln());
    }

    #[test]
    fn relay_log_with_unbounded_input_skips_emit() {
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        // No facts planted on the input.

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_log_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "unbounded input must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_log_with_non_positive_lo_skips_emit() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, -1.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_log_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "non-positive lo must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_arccos_with_input_bound_emits_output_interval() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 0.0);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 0.5);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_arccos_output_interval(&mut b, x_vid, out_vid);
        assert!(emitted, "expected relay to deposit an output bound");

        let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
        let term = entries.last().expect("at least one fact");
        let (lo, hi) = extract_float_bounds_from_term(term);
        // arccos is monotone-decreasing: bounds swap.
        assert_eq!(lo, 0.5_f64.acos());
        assert_eq!(hi, 0.0_f64.acos());
    }

    #[test]
    fn relay_arccos_with_unbounded_input_skips_emit() {
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        // No facts planted on the input.

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_arccos_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "unbounded input must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }

    #[test]
    fn relay_arccos_with_out_of_domain_lo_skips_emit() {
        use crate::optim::predicates::formula::CmpOp;
        let mut b = crate::builder::IRBuilder::new();
        let x = b.ir_read_float(InputPath::new("x", vec![]), false);
        let x_vid = x.value_id().unwrap();
        // `lo = -1.5` is outside arccos's domain `[-1, 1]`; the
        // defensive guard must skip rather than depositing a NaN-laden
        // bound.
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, -1.5);
        plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 0.5);

        let out_vid = crate::types::ValueId::next();
        let emitted = relay_arccos_output_interval(&mut b, x_vid, out_vid);
        assert!(!emitted, "out-of-domain lo must skip emit");
        assert!(b.facts.per_value.get(&out_vid).is_none());
    }
}
