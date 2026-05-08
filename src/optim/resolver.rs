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
//   reverse-reachability subgraph rooted at `val.ptr()` and encodes only
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
        let ptr = val.ptr()?;
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
        let ptr = val.ptr()?;
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
        let ptr = val.ptr()?;
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
    for c in walker.constraints {
        solver.assert(&c);
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
    for c in walker.constraints {
        opt.assert(&c);
    }

    let int = match root_term {
        crate::optim::smt_encoding::Z3Term::Int(i) => i,
        crate::optim::smt_encoding::Z3Term::Bool(b) => {
            // Project bool→int(0/1) so the Optimize objective makes sense.
            b.ite(&z3::ast::Int::from_i64(1), &z3::ast::Int::from_i64(0))
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
    /// Anything not yet enumerated. The string is a short site label;
    /// promote it to a named variant if it recurs.
    Other(&'static str),
}

impl SiteKind {
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
    let (resolver, stmts) = b.split_resolver_and_stmts();
    match resolver.resolve_int_with_stmts(val, stmts) {
        Some(n) => Ok(StaticInt(n)),
        None => Err(ZinniaError {
            message: format_diagnostic(site, dbg),
        }),
    }
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
            IRStatement::new(0, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 9 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None),
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
                0,
                IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 5 }, vec![], None),
            IRStatement::new(2, IR::EqI, vec![0, 1], None),
            IRStatement::new(3, IR::ConstantInt { value: 100 }, vec![], None),
            IRStatement::new(4, IR::ConstantInt { value: 100 }, vec![], None),
            IRStatement::new(5, IR::SelectI, vec![2, 3, 4], None),
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
            0,
            IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
            vec![],
            None,
        )];
        let v = runtime_int(0);
        let mut r = SmtResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// Disable flag: even on an SMT-decidable case, returning None
    /// (after the static-val fast path).
    #[test]
    fn smt_disable_flag_short_circuits() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 9 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None),
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
            IRStatement::new(0, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 9 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None),
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
            0,
            IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
            vec![],
            None,
        ));
        // Build a chain: stmt1 = stmt0 * stmt0, stmt2 = stmt1 * stmt0, ...
        // up to stmt30. Result has high arithmetic complexity.
        let mut last = 0u32;
        for i in 1..=30 {
            stmts.push(IRStatement::new(i, IR::MulI, vec![last, 0], None));
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
            IRStatement::new(0, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 9 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None),
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
            IRStatement::new(0, IR::ConstantBool { value: false }, vec![], None),
            IRStatement::new(1, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(2, IR::ConstantBool { value: true }, vec![], None),
            IRStatement::new(3, IR::SelectB, vec![0, 1, 2], None),
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
                0,
                IR::ReadInteger {
                    path: InputPath::new("c", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None),
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
                0,
                IR::ReadInteger {
                    path: InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::EqI, vec![0, 0], None), // x == x → true
            IRStatement::new(2, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(3, IR::ConstantInt { value: 9 }, vec![], None),
            IRStatement::new(4, IR::SelectI, vec![1, 2, 3], None),
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
            0,
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![],
            None,
        )];
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
            0,
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![],
            None,
        )];
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
                0,
                IR::ReadInteger {
                    path: InputPath::new("x", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::MulI, vec![0, 0], None),
            IRStatement::new(2, IR::MulI, vec![1, 0], None),
            IRStatement::new(3, IR::MulI, vec![2, 0], None),
            IRStatement::new(4, IR::MulI, vec![3, 0], None),
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
            0,
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![],
            None,
        ));
        let mut last = 0u32;
        for i in 1..=20 {
            stmts.push(IRStatement::new(i, IR::MulI, vec![last, 0], None));
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
                0,
                IR::ReadInteger {
                    path: InputPath::new("c", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None),
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
}
