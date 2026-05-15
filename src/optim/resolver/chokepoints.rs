//! Chokepoints: the typed "must-be-static" / "must-be-bounded" int helpers.
//!
//! Provides:
//! - [`StaticInt`] — the typed wrapper for a proven compile-time integer.
//! - [`SiteKind`] — diagnostic / telemetry tag for "must-be-static" sites.
//! - [`require_static_int`] — chokepoint that demands a unique compile-time
//!   integer (with fact-fallback).
//! - [`BoundedInt`] / [`require_static_or_bounded_int`] — bounded-aware
//!   companion.
//! - [`resolve_int_or_bounded`] / [`require_provable_static_int`] — adds the
//!   prove-based outward-doubling probe when the cheaper layers fail.

use crate::ast::DebugInfo;
use crate::error::ZinniaError;
use crate::types::Value;

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
/// [`super::StaticOnlyResolver`], tomorrow: SMT/range-augmented). Returns a
/// typed [`StaticInt`] on success, or a [`ZinniaError`] with a uniform
/// diagnostic referring to the [`SiteKind`].
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
/// Soundness: only [`crate::optim::prove::ProveOutcome::Proved`] admits
/// the bound. [`crate::optim::prove::ProveOutcome::Disproved`] and
/// [`crate::optim::prove::ProveOutcome::Unknown`] both default to "no
/// information" — the helper returns `Neither` for that half. Treating
/// `Unknown` as `Proved` would be a circuit-correctness bug.
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
