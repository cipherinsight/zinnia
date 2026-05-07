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
}
