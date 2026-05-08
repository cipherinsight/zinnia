//! P2 — `RangeResolver` and the supporting `IntInterval` domain.
//!
//! Phase 2 of `compiler.epic-restore-smt-reasoning`. Adds an interval
//! (range-analysis) layer that runs **before** SMT in the layered resolver
//! dispatch. Many queries — loop indices, modular reductions, clamped values —
//! resolve through cheap forward propagation alone, never paying SMT cost.
//!
//! See `kanban/cards/compiler/smt-range-analysis-pre-pass/README.md` for the
//! authoritative spec and design rationale.
//!
//! Module layout (filled in across phase-2 commits):
//!
//! 1. [`IntInterval`] + saturating combinators — this commit.
//! 2. [`RangeResolver`] scaffold + propagation for `ConstantInt`/`AddI`/`SubI`/
//!    `MulI`/`SelectI`/logicals — next commit.
//! 3. Modular / mask / clamp arms (`ModI`, `BitAndI`, `MinI`, `MaxI`, …).
//! 4. [`LayeredResolver`] composition (range → SMT) lives in `resolver.rs`.
//!
//! ## Soundness over precision
//!
//! Every combinator below uses **saturating** arithmetic (`i64::saturating_*`
//! or computed in `i128` and clamped at `i64::MIN..=i64::MAX`). Wrapping
//! arithmetic would be a soundness bug: if `MulI([i64::MAX, i64::MAX], [2, 2])`
//! wrapped, the resolver would later report a definitive integer value for an
//! expression that overflows at runtime, miscompiling the program. When in
//! doubt the combinators return [`IntInterval::unbounded`] — a wider interval
//! is always sound.

// ---------------------------------------------------------------------------
// IntInterval
// ---------------------------------------------------------------------------

/// Closed integer interval `[min, max]` over `i64`.
///
/// `i64::MIN` / `i64::MAX` represent "unbounded below / above" — see
/// [`IntInterval::unbounded`]. The invariant is `min <= max`; combinators that
/// would violate it (e.g., from unsatisfiable intersections) return
/// `unbounded()` instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntInterval {
    pub min: i64,
    pub max: i64,
}

impl IntInterval {
    /// A point interval `[n, n]`.
    pub fn point(n: i64) -> Self {
        Self { min: n, max: n }
    }

    /// The fully unbounded interval `[i64::MIN, i64::MAX]`. Returned by
    /// combinators when overflow / undefined behaviour / unknown operand
    /// makes a tighter result unsound.
    pub fn unbounded() -> Self {
        Self {
            min: i64::MIN,
            max: i64::MAX,
        }
    }

    /// `[0, 0]` — used for `ConstantBool { value: false }` and similar.
    pub fn zero() -> Self {
        Self::point(0)
    }

    /// `[0, 1]` — used for any boolean-projected-to-int wire.
    pub fn bool_domain() -> Self {
        Self { min: 0, max: 1 }
    }

    pub fn is_unbounded(&self) -> bool {
        self.min == i64::MIN && self.max == i64::MAX
    }

    pub fn is_point(&self) -> bool {
        self.min == self.max
    }

    pub fn contains(&self, n: i64) -> bool {
        self.min <= n && n <= self.max
    }

    // -----------------------------------------------------------------
    // Saturating combinators
    // -----------------------------------------------------------------

    /// Saturating addition. Overflow → unbounded endpoint.
    pub fn add(a: Self, b: Self) -> Self {
        Self {
            min: a.min.saturating_add(b.min),
            max: a.max.saturating_add(b.max),
        }
    }

    /// Saturating subtraction. `[a.min - b.max, a.max - b.min]`.
    pub fn sub(a: Self, b: Self) -> Self {
        Self {
            min: a.min.saturating_sub(b.max),
            max: a.max.saturating_sub(b.min),
        }
    }

    /// 4-corner product, saturating on overflow.
    pub fn mul(a: Self, b: Self) -> Self {
        let products = [
            sat_mul(a.min, b.min),
            sat_mul(a.min, b.max),
            sat_mul(a.max, b.min),
            sat_mul(a.max, b.max),
        ];
        let mut lo = products[0];
        let mut hi = products[0];
        for &p in &products[1..] {
            if p < lo {
                lo = p;
            }
            if p > hi {
                hi = p;
            }
        }
        Self { min: lo, max: hi }
    }

    /// Truncating integer division (`/` in Rust). If `b` straddles zero or
    /// touches zero we conservatively return unbounded — division by zero
    /// would be UB at runtime, and a divisor crossing zero produces
    /// arbitrarily large quotients near 0.
    pub fn div(a: Self, b: Self) -> Self {
        if b.contains(0) {
            return Self::unbounded();
        }
        // Both endpoints have the same sign; consider all 4 corners.
        let candidates = [
            sat_div_trunc(a.min, b.min),
            sat_div_trunc(a.min, b.max),
            sat_div_trunc(a.max, b.min),
            sat_div_trunc(a.max, b.max),
        ];
        let mut lo = candidates[0];
        let mut hi = candidates[0];
        for &c in &candidates[1..] {
            if c < lo {
                lo = c;
            }
            if c > hi {
                hi = c;
            }
        }
        Self { min: lo, max: hi }
    }

    /// Floor division (`//` in Python; matches the IR's `FloorDivI`).
    pub fn floor_div(a: Self, b: Self) -> Self {
        if b.contains(0) {
            return Self::unbounded();
        }
        let candidates = [
            sat_div_floor(a.min, b.min),
            sat_div_floor(a.min, b.max),
            sat_div_floor(a.max, b.min),
            sat_div_floor(a.max, b.max),
        ];
        let mut lo = candidates[0];
        let mut hi = candidates[0];
        for &c in &candidates[1..] {
            if c < lo {
                lo = c;
            }
            if c > hi {
                hi = c;
            }
        }
        Self { min: lo, max: hi }
    }

    /// Modulo. Conservative cases:
    /// - `b.min > 0` → result in `[0, min(b.max - 1, max(a.max, 0))]` when
    ///   `a >= 0`; in general `[0, b.max - 1]`.
    /// - `b.max < 0` → result in `[b.min + 1, 0]`.
    /// - `b` may be zero → unbounded (div-by-zero at runtime).
    pub fn modulo(a: Self, b: Self) -> Self {
        if b.contains(0) {
            return Self::unbounded();
        }
        if b.min > 0 {
            // Python-style mod: result is non-negative when divisor is
            // positive, regardless of dividend's sign. Bounded above by
            // `b.max - 1`.
            let upper = b.max.saturating_sub(1);
            // If we know dividend is small and non-negative, tighten.
            let cap = if a.min >= 0 && a.max < upper {
                a.max
            } else {
                upper
            };
            Self { min: 0, max: cap }
        } else {
            // b.max < 0. Python mod with negative divisor produces a
            // non-positive result.
            let lower = b.min.saturating_add(1);
            Self { min: lower, max: 0 }
        }
    }

    /// `[min(t.min,f.min), max(t.max,f.max)]` — both branches reachable.
    pub fn select(t: Self, f: Self) -> Self {
        Self {
            min: t.min.min(f.min),
            max: t.max.max(f.max),
        }
    }

    /// `[max(min), min(max)]` — narrow to the overlap. Used to model clamps
    /// and constant-equality refinements. If the intervals are disjoint
    /// (overlap empty), returns unbounded — soundness over precision.
    pub fn intersect(a: Self, b: Self) -> Self {
        let lo = a.min.max(b.min);
        let hi = a.max.min(b.max);
        if lo > hi {
            // Empty intersection — over-approximate.
            Self::unbounded()
        } else {
            Self { min: lo, max: hi }
        }
    }

    /// Bitwise AND mask analysis.
    ///
    /// When both operands are non-negative, the result is bounded above by
    /// the smaller of the two operand maxes (since `a & b <= min(a, b)`
    /// for non-negative integers). When either operand may be negative we
    /// fall back to unbounded — bitwise AND on signed two's-complement
    /// representations doesn't yield a clean numeric bound.
    pub fn bitand(a: Self, b: Self) -> Self {
        if a.min >= 0 && b.min >= 0 {
            // Non-negative: result in [0, min(a.max, b.max)].
            let upper = a.max.min(b.max);
            Self { min: 0, max: upper.max(0) }
        } else {
            Self::unbounded()
        }
    }

    /// Element-wise minimum.
    pub fn min_op(a: Self, b: Self) -> Self {
        Self {
            min: a.min.min(b.min),
            max: a.max.min(b.max),
        }
    }

    /// Element-wise maximum.
    pub fn max_op(a: Self, b: Self) -> Self {
        Self {
            min: a.min.max(b.min),
            max: a.max.max(b.max),
        }
    }
}

// ---------------------------------------------------------------------------
// i128-saturating helpers
// ---------------------------------------------------------------------------

/// Saturating absolute value: `i64::MIN.abs()` overflows, saturate to
/// `i64::MAX`.
fn sat_abs(a: i64) -> i64 {
    if a == i64::MIN {
        i64::MAX
    } else {
        a.abs()
    }
}

/// Saturating multiply `i64 * i64 -> i64`, computed in i128 and clamped.
fn sat_mul(a: i64, b: i64) -> i64 {
    let p = (a as i128) * (b as i128);
    if p > i64::MAX as i128 {
        i64::MAX
    } else if p < i64::MIN as i128 {
        i64::MIN
    } else {
        p as i64
    }
}

/// Truncating integer division, saturating on the `i64::MIN / -1` overflow.
/// Caller must ensure `b != 0`.
fn sat_div_trunc(a: i64, b: i64) -> i64 {
    if b == -1 && a == i64::MIN {
        // i64::MIN / -1 overflows.
        i64::MAX
    } else {
        a / b
    }
}

/// Python-style floor division, saturating on overflow. Caller must ensure
/// `b != 0`.
fn sat_div_floor(a: i64, b: i64) -> i64 {
    if b == -1 && a == i64::MIN {
        return i64::MAX;
    }
    let q = a / b;
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        // Truncation rounded toward zero but the true quotient is more
        // negative — adjust by one.
        q.saturating_sub(1)
    } else {
        q
    }
}

// ---------------------------------------------------------------------------
// RangeResolver
// ---------------------------------------------------------------------------
//
// Forward-walks the IR DAG to compute an [`IntInterval`] per ptr, caching
// results behind a `Mutex` so the outer `RangeResolver` can stay
// `Send + Sync` (required because `IRGraph` is held by a `#[pyclass]`).
//
// The cache holds intervals (not just resolved points) so that downstream
// queries on the same wire short-circuit, *and* so dependent ops can read
// the bounds of an unbounded-result wire without re-walking. `on_ir_mutated`
// blows the cache wholesale (P5 may refine).
//
// Why per-ptr is sound: each statement in the IR has a stable identity; an
// op's interval is determined by the intervals of its arguments which are
// themselves keyed by ptr. No path-condition refinement happens here (that's
// P5+ per spec).

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::optim::resolver::Resolver;
use crate::optim::telemetry::SmtTelemetry;
use crate::types::{StmtId, Value};

/// Range-analysis [`Resolver`]. See module-level comment for design.
#[derive(Debug)]
pub struct RangeResolver {
    cache: Mutex<HashMap<StmtId, IntInterval>>,
    /// P5 telemetry. Shared with the SMT layer when constructed via
    /// `LayeredResolver::with_telemetry` so a single summary reflects the
    /// whole pipeline.
    telemetry: Arc<SmtTelemetry>,
}

impl Default for RangeResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeResolver {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            telemetry: SmtTelemetry::new(),
        }
    }

    /// Swap in a shared telemetry handle (used by the layered constructor
    /// so range and SMT counters share one summary).
    pub fn with_telemetry(mut self, telemetry: Arc<SmtTelemetry>) -> Self {
        self.telemetry = telemetry;
        self
    }

    /// Borrow the telemetry handle.
    pub fn telemetry(&self) -> Arc<SmtTelemetry> {
        Arc::clone(&self.telemetry)
    }

    /// Test / telemetry helper: number of cached interval entries. P5 will
    /// surface this for the "how often did range answer first" metric.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Compute (and cache) the [`IntInterval`] for the wire at `ptr`. Walks
    /// arguments recursively. Unknown / off-the-int-path IR ops fall through
    /// to [`IntInterval::unbounded`], which is the conservative
    /// over-approximation.
    fn interval_of(&self, ptr: StmtId, stmts: &[IRStatement]) -> IntInterval {
        // Fast cache lookup.
        if let Some(cached) = self.cache.lock().unwrap().get(&ptr).copied() {
            return cached;
        }

        // Defensive bounds check — out-of-range ptr (shouldn't happen in
        // practice; the resolver was given the wrong stmt slice) yields the
        // safe over-approximation.
        let stmt = match stmts.get(ptr as usize) {
            Some(s) => s,
            None => return IntInterval::unbounded(),
        };

        let interval = match &stmt.ir {
            // Constants ------------------------------------------------
            IR::ConstantInt { value } => IntInterval::point(*value),
            IR::ConstantBool { value } => {
                IntInterval::point(if *value { 1 } else { 0 })
            }

            // Arithmetic -----------------------------------------------
            IR::AddI => self.binop(stmt, stmts, IntInterval::add),
            IR::SubI => self.binop(stmt, stmts, IntInterval::sub),
            IR::MulI => self.binop(stmt, stmts, IntInterval::mul),
            IR::DivI => self.binop(stmt, stmts, IntInterval::div),
            IR::FloorDivI => self.binop(stmt, stmts, IntInterval::floor_div),
            IR::ModI => self.binop(stmt, stmts, IntInterval::modulo),

            // Bitwise --------------------------------------------------
            IR::BitAndI => self.binop(stmt, stmts, IntInterval::bitand),
            IR::BitOrI => {
                // For non-negative operands, `a | b <= a + b` (no carry can
                // overshoot). Use saturating add as the upper bound.
                if stmt.arguments.len() == 2 {
                    let a = self.interval_of(stmt.arguments[0], stmts);
                    let b = self.interval_of(stmt.arguments[1], stmts);
                    if a.min >= 0 && b.min >= 0 {
                        IntInterval {
                            min: a.min.max(b.min),
                            max: a.max.saturating_add(b.max),
                        }
                    } else {
                        IntInterval::unbounded()
                    }
                } else {
                    IntInterval::unbounded()
                }
            }

            // Casts ----------------------------------------------------
            // bool→int: domain is [0, 1]. int→bool: also [0, 1] when
            // we ask for the int interval of the bool. Both go to
            // bool_domain.
            IR::IntCast => IntInterval::bool_domain(),
            IR::BoolCast => IntInterval::bool_domain(),

            // Unary integer arms ---------------------------------------
            // AbsI: result is [0, max(|min|, |max|)]; saturating handles
            // i64::MIN's |x|.
            IR::AbsI => {
                if stmt.arguments.len() == 1 {
                    let a = self.interval_of(stmt.arguments[0], stmts);
                    let lo = if a.contains(0) {
                        0
                    } else {
                        sat_abs(a.min).min(sat_abs(a.max))
                    };
                    let hi = sat_abs(a.min).max(sat_abs(a.max));
                    IntInterval { min: lo, max: hi }
                } else {
                    IntInterval::unbounded()
                }
            }
            // SignI returns -1, 0, or 1 — clamp to that domain.
            IR::SignI => {
                if stmt.arguments.len() == 1 {
                    let a = self.interval_of(stmt.arguments[0], stmts);
                    let lo = if a.min < 0 {
                        -1
                    } else if a.min == 0 {
                        0
                    } else {
                        1
                    };
                    let hi = if a.max > 0 {
                        1
                    } else if a.max == 0 {
                        0
                    } else {
                        -1
                    };
                    IntInterval { min: lo, max: hi }
                } else {
                    IntInterval::unbounded()
                }
            }

            // Selection (commit 2 covers SelectI) ----------------------
            IR::SelectI => {
                if stmt.arguments.len() == 3 {
                    let t = self.interval_of(stmt.arguments[1], stmts);
                    let f = self.interval_of(stmt.arguments[2], stmts);
                    IntInterval::select(t, f)
                } else {
                    IntInterval::unbounded()
                }
            }

            // Logical: always boolean-projected → [0, 1] ----------------
            IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot => {
                IntInterval::bool_domain()
            }

            // Comparisons project to bool: [0, 1] -----------------------
            IR::EqI
            | IR::NeI
            | IR::LtI
            | IR::LteI
            | IR::GtI
            | IR::GteI
            | IR::EqF
            | IR::NeF
            | IR::LtF
            | IR::LteF
            | IR::GtF
            | IR::GteF
            | IR::EqHash => IntInterval::bool_domain(),

            // SelectB returns a bool; project to [0, 1].
            IR::SelectB => IntInterval::bool_domain(),

            // Everything else falls through to commit 3 / unbounded.
            _ => IntInterval::unbounded(),
        };

        self.cache.lock().unwrap().insert(ptr, interval);
        interval
    }

    /// Helper to compute a binary op's interval given its two arguments.
    fn binop(
        &self,
        stmt: &IRStatement,
        stmts: &[IRStatement],
        f: fn(IntInterval, IntInterval) -> IntInterval,
    ) -> IntInterval {
        if stmt.arguments.len() != 2 {
            return IntInterval::unbounded();
        }
        let a = self.interval_of(stmt.arguments[0], stmts);
        let b = self.interval_of(stmt.arguments[1], stmts);
        f(a, b)
    }
}

impl Resolver for RangeResolver {
    /// Without `&[IRStatement]` we can't walk the DAG; fall back to
    /// `static_val`. Call sites should route through
    /// `resolve_int_with_stmts` (the IRBuilder/IRGraph split-borrow helper
    /// already does this for the `require_static_int` chokepoint).
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
        // Fast path 1: static-val.
        if let Some(n) = val.int_val() {
            return Some(n);
        }
        // Fast path 2: walk the interval and report a point if we got one.
        let ptr = val.ptr()?;
        let t0 = Instant::now();
        let interval = self.interval_of(ptr, stmts);
        self.telemetry.record_range_duration(t0.elapsed());
        if interval.is_point() {
            self.telemetry.queries_range_hit.fetch_add(1, Ordering::Relaxed);
            Some(interval.min)
        } else {
            None
        }
    }

    /// P2 is integer-only — we don't refine booleans through interval
    /// analysis. Call sites that need bool resolution fall through to a
    /// later layer (SmtResolver in the layered composition). One trivial
    /// shortcut: when both ints are point-and-equal we can return that
    /// projected to bool — but that's already covered by `resolve_int`'s
    /// `[0, 1]` arms returning `Some(0)` or `Some(1)`. We mirror it here
    /// to keep the bool API symmetric.
    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<bool> {
        if let Some(b) = val.bool_val() {
            return Some(b);
        }
        let ptr = val.ptr()?;
        let t0 = Instant::now();
        let interval = self.interval_of(ptr, stmts);
        self.telemetry.record_range_duration(t0.elapsed());
        // Only resolve to bool if the wire is a [0, 1]-domain wire that
        // collapsed to a single point.
        if interval.is_point() && (interval.min == 0 || interval.min == 1) {
            self.telemetry.queries_range_hit.fetch_add(1, Ordering::Relaxed);
            Some(interval.min == 1)
        } else {
            None
        }
    }

    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        if let Some(n) = val.int_val() {
            return Some(n);
        }
        let ptr = val.ptr()?;
        let t0 = Instant::now();
        let interval = self.interval_of(ptr, stmts);
        self.telemetry.record_range_duration(t0.elapsed());
        if interval.max == i64::MAX {
            None
        } else {
            self.telemetry.queries_range_hit.fetch_add(1, Ordering::Relaxed);
            Some(interval.max)
        }
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
        let t0 = Instant::now();
        let interval = self.interval_of(ptr, stmts);
        self.telemetry.record_range_duration(t0.elapsed());
        if interval.min == i64::MIN {
            None
        } else {
            self.telemetry.queries_range_hit.fetch_add(1, Ordering::Relaxed);
            Some(interval.min)
        }
    }

    fn on_ir_mutated(&mut self, _affected: &[StmtId]) {
        // P2 conservative: blow the cache. P5 may refine to precise ids.
        self.cache.lock().unwrap().clear();
    }

    fn telemetry_handle(
        &self,
    ) -> Option<std::sync::Arc<crate::optim::telemetry::SmtTelemetry>> {
        Some(Arc::clone(&self.telemetry))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(min: i64, max: i64) -> IntInterval {
        IntInterval { min, max }
    }

    #[test]
    fn add_basic() {
        assert_eq!(IntInterval::add(iv(2, 5), iv(3, 4)), iv(5, 9));
    }

    #[test]
    fn sub_basic() {
        // [a.min - b.max, a.max - b.min]
        assert_eq!(IntInterval::sub(iv(10, 20), iv(3, 5)), iv(5, 17));
    }

    #[test]
    fn mul_4_corner_with_negatives() {
        // [-2, 3] * [-1, 4]: corners are 2, -8, -3, 12. Result is [-8, 12].
        assert_eq!(IntInterval::mul(iv(-2, 3), iv(-1, 4)), iv(-8, 12));
    }

    #[test]
    fn mul_saturates_on_overflow() {
        // i64::MAX * 2 overflows; must saturate, NEVER wrap.
        let r = IntInterval::mul(iv(i64::MAX, i64::MAX), iv(2, 2));
        assert_eq!(r, iv(i64::MAX, i64::MAX));
    }

    #[test]
    fn div_zero_crossing_returns_unbounded() {
        // Divisor crosses zero — result is unbounded.
        assert_eq!(IntInterval::div(iv(10, 20), iv(-5, 5)), IntInterval::unbounded());
    }

    #[test]
    fn modulo_positive_divisor_bounds_result() {
        // i in [0, 100], j in [1, 8] → i % j in [0, 7].
        assert_eq!(IntInterval::modulo(iv(0, 100), iv(1, 8)), iv(0, 7));
    }

    #[test]
    fn bitand_mask_analysis() {
        // [0, 100] & [0, 7] → [0, 7] (mask analysis on the smaller operand).
        assert_eq!(IntInterval::bitand(iv(0, 100), iv(0, 7)), iv(0, 7));
    }

    #[test]
    fn intersect_disjoint_is_unbounded() {
        // [0, 5] ∩ [10, 20] = empty → over-approximate to unbounded.
        assert_eq!(
            IntInterval::intersect(iv(0, 5), iv(10, 20)),
            IntInterval::unbounded()
        );
    }

    // -----------------------------------------------------------------
    // RangeResolver tests (commit 2 — cheap arms)
    // -----------------------------------------------------------------

    use crate::circuit_input::InputPath;
    use crate::types::ScalarValue;

    /// Helper: Value::Integer with runtime-only ptr.
    fn runtime_int(stmt_id: StmtId) -> Value {
        Value::Integer(ScalarValue::runtime(stmt_id))
    }

    /// `select(c, 7, 7)` collapses to `7` regardless of `c` — this is the
    /// canonical case the static_val path can't handle but range can: both
    /// branches have the same point interval, so the union is also a
    /// point.
    #[test]
    fn range_select_unifies_branches() {
        // stmt0 = ReadInteger("c") (free cond, here projected to int via
        //          a comparison… but for this test we just need a free
        //          ptr at index 0). For simplicity we use ReadInteger as
        //          the "cond" placeholder; the resolver doesn't read it.
        // stmt1 = ConstantInt(7), stmt2 = ConstantInt(7),
        // stmt3 = SelectI(stmt0, stmt1, stmt2).
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
        let mut r = RangeResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(7));
    }

    /// `AddI(free, free)` where both operands are unbounded — the sum is
    /// unbounded and resolve_int returns None.
    #[test]
    fn range_unbounded_returns_none() {
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
            IRStatement::new(
                1,
                IR::ReadInteger {
                    path: InputPath::new("y", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
        ];
        let v = runtime_int(2);
        let mut r = RangeResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
        // resolve_max / resolve_min on the unbounded result also return
        // None (we don't fabricate i64::MAX as an answer).
        assert_eq!(r.resolve_max_with_stmts(&v, &stmts), None);
        assert_eq!(r.resolve_min_with_stmts(&v, &stmts), None);
    }

    /// `MulI([2,2], [3,5])` — `[2,2]` is a constant, `[3,5]` is the union
    /// of two select branches. Range computes max=10, min=6.
    #[test]
    fn range_resolve_max_returns_endpoint() {
        // Build a wire whose interval is [3, 5] via `select(c, 3, 5)`.
        // Then multiply by 2 (constant) and ask for resolve_max.
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
            IRStatement::new(1, IR::ConstantInt { value: 3 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 5 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None), // [3, 5]
            IRStatement::new(4, IR::ConstantInt { value: 2 }, vec![], None),
            IRStatement::new(5, IR::MulI, vec![3, 4], None), // [6, 10]
        ];
        let v = runtime_int(5);
        let mut r = RangeResolver::new();
        assert_eq!(r.resolve_max_with_stmts(&v, &stmts), Some(10));
        assert_eq!(r.resolve_min_with_stmts(&v, &stmts), Some(6));
        // Not a point, so resolve_int still returns None.
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// `AddI(constant 3, constant 4)` — both operands collapse to point
    /// intervals and the sum is a point. resolve_int returns Some(7).
    /// (static_val could fold this when constructed via the IRBuilder, but
    /// here we hand-build IRStatements without static_val on the wire, so
    /// only the range walker proves it.)
    #[test]
    fn range_resolves_constant_add() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 3 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 4 }, vec![], None),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
        ];
        // Build a Value::Integer with no static_val but ptr=2.
        let v = runtime_int(2);
        let mut r = RangeResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(7));
    }

    /// Calling resolve twice on the same wire should hit the cache the
    /// second time. We verify by counting cache entries.
    #[test]
    fn range_caches_intervals() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 3 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 4 }, vec![], None),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
        ];
        let v = runtime_int(2);
        let mut r = RangeResolver::new();
        assert_eq!(r.cache_size(), 0);
        let _ = r.resolve_int_with_stmts(&v, &stmts);
        // Cache holds intervals for stmts 0, 1, 2.
        assert_eq!(r.cache_size(), 3);
        let _ = r.resolve_int_with_stmts(&v, &stmts);
        // No new entries on the second call.
        assert_eq!(r.cache_size(), 3);
    }

    /// `on_ir_mutated` clears the cache (P2 conservative).
    #[test]
    fn range_on_ir_mutated_clears_cache() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 3 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 4 }, vec![], None),
            IRStatement::new(2, IR::AddI, vec![0, 1], None),
        ];
        let v = runtime_int(2);
        let mut r = RangeResolver::new();
        let _ = r.resolve_int_with_stmts(&v, &stmts);
        assert!(r.cache_size() > 0);
        r.on_ir_mutated(&[]);
        assert_eq!(r.cache_size(), 0);
    }

    /// Logical / comparison ops are bool-projected: result interval is
    /// always [0, 1], so resolve_max == 1 and resolve_min == 0.
    #[test]
    fn range_bool_projected_ops_are_zero_one() {
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
            IRStatement::new(
                1,
                IR::ReadInteger {
                    path: InputPath::new("y", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(2, IR::LtI, vec![0, 1], None),
        ];
        // The result of LtI is a Boolean wire; range projects to [0, 1].
        let v = Value::Boolean(ScalarValue::runtime(2));
        let mut r = RangeResolver::new();
        // We're going to ask via the int interface here (the bool wire is
        // bound to a ptr; resolve_max_with_stmts is the public entry).
        assert_eq!(r.resolve_max_with_stmts(&v, &stmts), Some(1));
        assert_eq!(r.resolve_min_with_stmts(&v, &stmts), Some(0));
    }

    /// RangeResolver is `Send + Sync` (required by the Resolver trait).
    #[test]
    fn range_resolver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RangeResolver>();
    }

    // -----------------------------------------------------------------
    // RangeResolver tests (commit 3 — modular / mask / clamp arms)
    // -----------------------------------------------------------------

    /// The headline win for range analysis: `(i * 7) % 64` with
    /// `i ∈ [0, 63]` — this is the canonical loop-index-into-mod-table
    /// pattern. SMT *can* prove the result is in [0, 63] but pays
    /// milliseconds; range proves it for free, so the layered resolver
    /// (range → SMT) skips the Z3 call entirely. We model
    /// `i ∈ [0, 63]` via `select(c, 0, 63)` since Range can't yet ingest
    /// arbitrary path conditions; the test still demonstrates the
    /// cascading propagation through MulI then ModI.
    #[test]
    fn range_modular_index_pattern() {
        let stmts = vec![
            // stmt0..3: build i ∈ [0, 63] via select(c, 0, 63).
            IRStatement::new(
                0,
                IR::ReadInteger {
                    path: InputPath::new("c", vec![]),
                    is_public: false,
                },
                vec![],
                None,
            ),
            IRStatement::new(1, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 63 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None), // [0, 63]
            // stmt4: 7. stmt5: i * 7 → [0, 441].
            IRStatement::new(4, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(5, IR::MulI, vec![3, 4], None),
            // stmt6: 64. stmt7: (i*7) % 64 → [0, 63].
            IRStatement::new(6, IR::ConstantInt { value: 64 }, vec![], None),
            IRStatement::new(7, IR::ModI, vec![5, 6], None),
        ];
        let v = runtime_int(7);
        let mut r = RangeResolver::new();
        // Result is [0, 63] — bounded above by 63, below by 0.
        assert_eq!(r.resolve_max_with_stmts(&v, &stmts), Some(63));
        assert_eq!(r.resolve_min_with_stmts(&v, &stmts), Some(0));
        // Not a point, so resolve_int returns None.
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    }

    /// `BitAndI([0, 100], 7)` — mask analysis: result bounded by 7.
    #[test]
    fn range_bitand_mask_bounds_result() {
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
            IRStatement::new(1, IR::ConstantInt { value: 0 }, vec![], None),
            IRStatement::new(2, IR::ConstantInt { value: 100 }, vec![], None),
            IRStatement::new(3, IR::SelectI, vec![0, 1, 2], None), // [0, 100]
            IRStatement::new(4, IR::ConstantInt { value: 7 }, vec![], None),
            IRStatement::new(5, IR::BitAndI, vec![3, 4], None), // [0, 7]
        ];
        let v = runtime_int(5);
        let mut r = RangeResolver::new();
        assert_eq!(r.resolve_max_with_stmts(&v, &stmts), Some(7));
        assert_eq!(r.resolve_min_with_stmts(&v, &stmts), Some(0));
    }

    /// `FloorDivI(constant 16, constant 4)` resolves to a point. Verifies
    /// the floor-div arm wires through; with both operands constant, the
    /// 4-corner endpoints all collapse to 4.
    #[test]
    fn range_floor_div_constant_resolves() {
        let stmts = vec![
            IRStatement::new(0, IR::ConstantInt { value: 16 }, vec![], None),
            IRStatement::new(1, IR::ConstantInt { value: 4 }, vec![], None),
            IRStatement::new(2, IR::FloorDivI, vec![0, 1], None), // [4, 4]
        ];
        let v = runtime_int(2);
        let mut r = RangeResolver::new();
        assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(4));
    }
}
