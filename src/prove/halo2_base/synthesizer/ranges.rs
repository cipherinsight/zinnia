//! Constraint-proven value ranges, and the comparisons and range assertions
//! built on them.
//!
//! A *proven range* of a cell is an interval that the constraints already
//! emitted force the integer the cell encodes into: a range check, an
//! asserted bound, a constant, a boolean gate, or a gate applied to cells
//! with proven ranges. Only constraint facts qualify, never witness values,
//! so the decisions taken from them are the same at key generation and at
//! proving. Provenance intervals from the IR builder are *not* proven
//! ranges: they only narrow the choice sets of memory reads, where a wrong
//! interval makes the circuit unsatisfiable rather than unsound.
//!
//! Comparisons use the ranges to size their check: two operands whose
//! difference is known to lie within `2^n` of zero are compared with one
//! `n`-bit range check (see [`Halo2BaseSynthesizer::less_than`]).

use halo2_base::gates::{GateInstructions, RangeInstructions};
use halo2_base::QuantumCell;

use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::prove::traits::Synthesizer;

/// Proven ranges are kept within `±2^LIMIT_BITS`, far below `p / 2`, so
/// interval arithmetic on them never meets the field's wrap-around.
const LIMIT_BITS: u32 = 120;

/// An inclusive interval of integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Interval {
    pub lo: i128,
    pub hi: i128,
}

impl Interval {
    /// The signed 64-bit window `[-2^63, 2^63)`, where Zinnia's int64 and
    /// raw Q32 values live.
    pub const SIGNED64: Interval = Interval { lo: -(1 << 63), hi: (1 << 63) - 1 };
    /// `[0, 2^64)`, the operand domain of the IR's unsigned comparisons.
    pub const UNSIGNED64: Interval = Interval { lo: 0, hi: (1 << 64) - 1 };
    pub const BIT: Interval = Interval { lo: 0, hi: 1 };

    pub fn new(lo: i128, hi: i128) -> Option<Interval> {
        let limit = 1i128 << LIMIT_BITS;
        (lo <= hi && lo >= -limit && hi <= limit).then_some(Interval { lo, hi })
    }

    pub fn point(v: i128) -> Option<Interval> {
        Interval::new(v, v)
    }

    /// `[0, 2^bits)`.
    pub fn bits(bits: usize) -> Option<Interval> {
        (bits < LIMIT_BITS as usize).then(|| Interval { lo: 0, hi: (1i128 << bits) - 1 })
    }

    pub fn within(&self, outer: &Interval) -> bool {
        outer.lo <= self.lo && self.hi <= outer.hi
    }

    pub fn hull(&self, other: &Interval) -> Interval {
        Interval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Interval) -> Option<Interval> {
        Interval::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    pub fn add(&self, other: &Interval) -> Option<Interval> {
        Interval::new(self.lo.checked_add(other.lo)?, self.hi.checked_add(other.hi)?)
    }

    pub fn sub(&self, other: &Interval) -> Option<Interval> {
        Interval::new(self.lo.checked_sub(other.hi)?, self.hi.checked_sub(other.lo)?)
    }

    pub fn neg(&self) -> Interval {
        Interval { lo: -self.hi, hi: -self.lo }
    }

    pub fn mul(&self, other: &Interval) -> Option<Interval> {
        let products = [
            self.lo.checked_mul(other.lo)?,
            self.lo.checked_mul(other.hi)?,
            self.hi.checked_mul(other.lo)?,
            self.hi.checked_mul(other.hi)?,
        ];
        Interval::new(*products.iter().min()?, *products.iter().max()?)
    }

    pub fn offset(&self, c: i128) -> Option<Interval> {
        Interval::new(self.lo.checked_add(c)?, self.hi.checked_add(c)?)
    }
}

/// A cell known to equal `bit ? if_one : if_zero` for a bit cell `bit`
/// (a select between constants, a bit times a constant, a masked bit).
/// Comparisons and equality tests against a constant then reduce to the
/// bit, its negation, or a constant.
#[derive(Clone, Copy)]
pub(super) struct TwoValued {
    pub bit: Halo2BaseCell,
    pub if_zero: i128,
    pub if_one: i128,
}

impl TwoValued {
    fn map(&self, f: impl Fn(i128) -> Option<i128>) -> Option<TwoValued> {
        Some(TwoValued { bit: self.bit, if_zero: f(self.if_zero)?, if_one: f(self.if_one)? })
    }

    fn hull(&self) -> Option<Interval> {
        Interval::new(self.if_zero.min(self.if_one), self.if_zero.max(self.if_one))
    }
}

/// A comparison bit's meaning: `[x < bound]` (`x_below`) or `[bound < x]`.
#[derive(Clone, Copy)]
pub(super) struct Comparison {
    x: Halo2BaseCell,
    bound: i128,
    x_below: bool,
}

/// An int64 window check of `x`: `shifted = x + 2^63` range-checked to 64
/// bits, with the top limb of that check (`top_bits` wide, in
/// `[0, 2^top_bits)` by the check's own lookups). See
/// `Halo2BaseSynthesizer::sign_from_window`.
#[derive(Clone, Copy)]
pub(super) struct WindowCheck {
    pub(super) shifted: Halo2BaseCell,
    pub(super) top_limb: Halo2BaseCell,
    pub(super) top_bits: usize,
}

/// The number of bits `n` with `2^n > v`, for `v >= 0`.
fn bits_above(v: i128) -> usize {
    (128 - v.max(0).leading_zeros()) as usize
}

/// An operand of a comparison: a cell or a constant.
#[derive(Clone, Copy)]
pub(super) enum Operand {
    Cell(Halo2BaseCell),
    Constant(i128),
}

impl Operand {
    fn quantum(&self) -> QuantumCell<Fp> {
        match self {
            Operand::Cell(c) => c.quantum(),
            Operand::Constant(v) => QuantumCell::Constant(kernel::i128_to_fp(*v)),
        }
    }

    fn value(&self) -> Fp {
        match self {
            Operand::Cell(c) => c.value,
            Operand::Constant(v) => kernel::i128_to_fp(*v),
        }
    }
}

impl Halo2BaseSynthesizer<'_> {
    /// The range the constraints force `x` into, if one is known.
    pub(super) fn proven_range(&self, x: &Halo2BaseCell) -> Option<Interval> {
        if self.is_const_cell(x) {
            return kernel::fp_to_i128(x.value).and_then(Interval::point);
        }
        let known = self.proven.get(&x.id).copied();
        if self.bit_cell_ids.contains(&x.id) {
            return Some(known.and_then(|iv| iv.intersect(&Interval::BIT)).unwrap_or(Interval::BIT));
        }
        known
    }

    /// Record that the constraints force `x` into `range` (intersected with
    /// what is already known). Callers state the constraint next to the call.
    ///
    /// When `x` is a cell `y + c` for a constant `c` (`record_offset`), `y` is
    /// forced into `range - c` too: the gate makes `y = x - c` in the field,
    /// and with `x` in `range` (within `±2^120`) and `|c| < 2^64` that is
    /// the integer `x - c`, far from wrapping. (A merged range check of
    /// `x - lo` so proves `x` in `[lo, hi]`, `checks::range_merges`.)
    pub(super) fn prove_range(&mut self, x: &Halo2BaseCell, range: Interval) {
        let range = match self.proven.get(&x.id) {
            Some(known) => known.intersect(&range).unwrap_or(range),
            None => range,
        };
        self.proven.insert(x.id, range);
        if let Some((base, c)) = self.offsets.get(&x.id).copied() {
            if let Some(shifted) = range.offset(-c) {
                if self.proven.get(&base.id).is_none_or(|known| !known.within(&shifted)) {
                    self.prove_range(&base, shifted);
                }
            }
        }
    }

    /// Record `out = x + sign * c` when `c` is a constant cell (`|c| < 2^64`),
    /// so a range later proven for `out` is carried back to `x`.
    pub(super) fn record_offset(&mut self, out: &Halo2BaseCell, x: &Halo2BaseCell, c: &Halo2BaseCell, sign: i128) {
        if self.is_const_cell(x) || !self.is_const_cell(c) {
            return;
        }
        if let Some(c) = kernel::fp_to_i128(c.value).filter(|c| c.unsigned_abs() < 1u128 << 64) {
            self.offsets.insert(out.id, (*x, sign * c));
        }
    }

    /// Record `out`'s range from its operands' ranges, when both are known.
    pub(super) fn propagate_range(
        &mut self,
        out: &Halo2BaseCell,
        a: &Halo2BaseCell,
        b: &Halo2BaseCell,
        combine: impl FnOnce(&Interval, &Interval) -> Option<Interval>,
    ) {
        if let (Some(ra), Some(rb)) = (self.proven_range(a), self.proven_range(b)) {
            if let Some(range) = combine(&ra, &rb) {
                self.prove_range(out, range);
            }
        }
    }

    /// `x` as `bit ? if_one : if_zero`, if the constraints make it so.
    pub(super) fn two_valued_of(&self, x: &Halo2BaseCell) -> Option<TwoValued> {
        if self.bit_cell_ids.contains(&x.id) {
            return Some(TwoValued { bit: *x, if_zero: 0, if_one: 1 });
        }
        self.two_valued.get(&x.id).copied()
    }

    /// Record that the gates force `x = tv.bit ? tv.if_one : tv.if_zero`
    /// (and with it the range of the two values).
    pub(super) fn set_two_valued(&mut self, x: &Halo2BaseCell, tv: TwoValued) {
        if tv.if_zero == tv.if_one {
            return;
        }
        if let Some(range) = tv.hull() {
            self.prove_range(x, range);
            self.two_valued.insert(x.id, tv);
        }
    }

    /// Two-valued facts through `out = f(x, c)` for a constant operand `c`
    /// (sums, differences and products with a constant).
    pub(super) fn propagate_two_valued(
        &mut self,
        out: &Halo2BaseCell,
        x: &Halo2BaseCell,
        c: &Halo2BaseCell,
        f: impl Fn(i128, i128) -> Option<i128>,
    ) {
        let constant = self.is_const_cell(c).then(|| kernel::fp_to_i128(c.value)).flatten();
        if let (Some(tv), Some(c)) = (self.two_valued_of(x), constant) {
            if let Some(mapped) = tv.map(|v| f(v, c)) {
                self.set_two_valued(out, mapped);
            }
        }
    }

    /// `x < c` (or `c < x` with `constant_on_left`) for a two-valued `x`
    /// and a constant `c`: the constant result, the bit, or its negation.
    fn decide_two_valued(&mut self, tv: TwoValued, c: i128, constant_on_left: bool) -> Result<Halo2BaseCell, ProvingError> {
        let holds = |v: i128| if constant_on_left { c < v } else { v < c };
        match (holds(tv.if_zero), holds(tv.if_one)) {
            (z, o) if z == o => self.constant_bool(z),
            (false, true) => Ok(tv.bit),
            _ => self.logical_not(&tv.bit),
        }
    }

    /// `[x == c]` for a two-valued `x` and a constant `c`, if decidable that
    /// way (always, as the two values differ).
    pub(super) fn equals_constant(&mut self, x: &Halo2BaseCell, c: &Halo2BaseCell) -> Result<Option<Halo2BaseCell>, ProvingError> {
        let constant = self.is_const_cell(c).then(|| kernel::fp_to_i128(c.value)).flatten();
        let (Some(tv), Some(c)) = (self.two_valued_of(x), constant) else {
            return Ok(None);
        };
        Ok(Some(if c == tv.if_one {
            tv.bit
        } else if c == tv.if_zero {
            self.logical_not(&tv.bit)?
        } else {
            self.constant_bool(false)?
        }))
    }

    fn operand_range(&self, x: &Operand) -> Option<Interval> {
        match x {
            Operand::Cell(c) => self.proven_range(c),
            Operand::Constant(v) => Interval::point(*v),
        }
    }

    /// Advice plus lookup cells of a `bits`-bit range check, following
    /// halo2-base's `range_check` layout: `n` lookup-width limbs recombined
    /// by an inner product (`3n - 2` cells, none for one limb), plus a
    /// shifted copy of a partial top limb (a bit check for a 1-bit one).
    pub(super) fn range_check_cost(&mut self, bits: usize) -> usize {
        let lookup = self.range_chip().lookup_bits();
        let limbs = bits.div_ceil(lookup).max(1);
        let recombine = if limbs == 1 { 0 } else { 3 * limbs - 2 };
        let partial = match bits % lookup {
            0 => 0,
            1 => 4,
            _ => 5,
        };
        recombine + limbs + partial
    }

    /// Constrain `0 <= x < 2^bits` (and record it).
    pub(super) fn range_check(&mut self, x: &Halo2BaseCell, bits: usize) {
        let x_cell = self.assigned(x);
        let range = self.range_chip();
        range.range_check(self.ctx(), x_cell, bits);
        if let Some(iv) = Interval::bits(bits) {
            self.prove_range(x, iv);
        }
    }

    /// Constrain `x` into `domain` (`SIGNED64` or `UNSIGNED64`) unless its
    /// proven range already lies inside: one 64-bit range check, of `x + 2^63`
    /// for the signed window.
    pub(super) fn ensure_within(&mut self, x: &Halo2BaseCell, domain: Interval) -> Result<(), ProvingError> {
        if self.proven_range(x).is_some_and(|r| r.within(&domain)) {
            return Ok(());
        }
        // A bound `x < hi` the circuit asserts anyway (see `plan.rs`), emitted
        // here: it proves `x` in `[0, hi)`, inside the domain, more cheaply.
        if let Some(hi) = self.planned_bound(x).filter(|hi| Interval::new(0, hi - 1).is_some_and(|r| r.within(&domain))) {
            return self.assert_less(x, Operand::Constant(hi));
        }
        if domain == Interval::UNSIGNED64 {
            self.range_check(x, 64);
        } else if domain == Interval::SIGNED64 {
            let shift = Fp::from(1u64 << 63);
            let shifted = self.gate_cell(x.value + shift, |gate, ctx| {
                gate.add(ctx, x.quantum(), QuantumCell::Constant(shift))
            });
            if let Some(window) = self.window_range_check(&shifted) {
                self.windows.insert(x.id, window);
            }
        } else {
            return Err(ProvingError::synthesis("unsupported comparison domain"));
        }
        self.prove_range(x, domain);
        Ok(())
    }

    /// The constant bound `hi` of an `AssertU128Lt(x, hi)` statement of the
    /// circuit on `x`'s IR value, if there is one.
    fn planned_bound(&self, x: &Halo2BaseCell) -> Option<i128> {
        let vid = self.vid_of_cell.get(&x.id)?;
        self.plan.bounds.get(vid).copied()
    }

    /// Constrain a hinted quotient `q` so that its division identity holds
    /// over the integers and has one solution: into `implied` (the range the
    /// dividend's proven range and the divisor allow) when that lies inside
    /// the signed 64-bit window and its check is cheaper, else into the
    /// window. Either way `|q|` stays below `2^64`, which is all the callers'
    /// uniqueness arguments need, and the honest quotient satisfies the check.
    pub(super) fn bound_quotient(&mut self, q: &Halo2BaseCell, implied: Option<Interval>) -> Result<(), ProvingError> {
        let narrow = implied.filter(|r| r.within(&Interval::SIGNED64)).filter(|r| {
            let width = self.lookup_aligned(bits_above(r.hi - r.lo).max(1));
            width < 64
        });
        match narrow {
            Some(range) => {
                let width = self.lookup_aligned(bits_above(range.hi - range.lo).max(1));
                let shifted = if range.lo == 0 {
                    *q
                } else {
                    let lo = kernel::i128_to_fp(range.lo);
                    self.gate_cell(q.value - lo, |gate, ctx| gate.sub(ctx, q.quantum(), QuantumCell::Constant(lo)))
                };
                self.range_check(&shifted, width);
                if let Some(checked) = Interval::new(range.lo, range.lo + (1i128 << width) - 1) {
                    self.prove_range(q, checked);
                }
                Ok(())
            }
            None => self.ensure_signed64(q),
        }
    }

    /// `a - b` (one gate, cached per operand pair so that a comparison and
    /// the selects on its result share it), with the range propagated.
    pub(super) fn difference(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Halo2BaseCell {
        if let Some(diff) = self.differences.get(&(a.id, b.id)) {
            return *diff;
        }
        let out = self.gate_cell(a.value - b.value, |gate, ctx| gate.sub(ctx, a.quantum(), b.quantum()));
        self.propagate_range(&out, a, b, Interval::sub);
        self.differences.insert((a.id, b.id), out);
        out
    }

    /// `x + c` for a constant `c` (one gate), with `x`'s range and
    /// two-valued facts shifted.
    pub(super) fn add_constant(&mut self, x: &Halo2BaseCell, c: i128) -> Halo2BaseCell {
        let c_fp = kernel::i128_to_fp(c);
        let out = self.gate_cell(x.value + c_fp, |gate, ctx| gate.add(ctx, x.quantum(), QuantumCell::Constant(c_fp)));
        if let Some(range) = self.proven_range(x).and_then(|r| r.offset(c)) {
            self.prove_range(&out, range);
        }
        if let Some(tv) = self.two_valued_of(x).and_then(|tv| tv.map(|v| v.checked_add(c))) {
            self.set_two_valued(&out, tv);
        }
        out
    }

    /// `c - x` for a constant `c` (one gate), with the range carried over.
    pub(super) fn constant_minus(&mut self, c: i128, x: &Halo2BaseCell) -> Halo2BaseCell {
        let c_fp = kernel::i128_to_fp(c);
        let out = self.gate_cell(c_fp - x.value, |gate, ctx| gate.sub(ctx, QuantumCell::Constant(c_fp), x.quantum()));
        if let Some(range) = self.proven_range(x).and_then(|r| r.neg().offset(c)) {
            self.prove_range(&out, range);
        }
        out
    }

    pub(super) fn ensure_signed64(&mut self, x: &Halo2BaseCell) -> Result<(), ProvingError> {
        self.ensure_within(x, Interval::SIGNED64)
    }

    fn ensure_operand_within(&mut self, x: &Operand, domain: Interval) -> Result<(), ProvingError> {
        match x {
            Operand::Cell(c) => self.ensure_within(c, domain),
            Operand::Constant(v) if domain.lo <= *v && *v <= domain.hi => Ok(()),
            Operand::Constant(_) => Err(ProvingError::synthesis("comparison constant outside the 64-bit window")),
        }
    }

    /// `[a < b]` as a bit. With `Some(domain)` each operand is constrained
    /// into `domain` (range-checked once unless already known to lie inside);
    /// with `None` both must already have proven ranges (internal values).
    ///
    /// With `D = A - B` proven to lie in `[d_lo, d_hi]` and `2^n` above both
    /// `d_hi` and `-d_lo`: the hint `s` is a bit, and `t = (a - b) + s·2^n`
    /// is range-checked to `[0, 2^n)`. For `s = 0` that is `0 <= D < 2^n`,
    /// for `s = 1` it is `-2^n <= D < 0`, so `s = [A < B]` is the only
    /// admissible value (`|t| < 2^(n+1)`, far below `p / 2`, so the field
    /// value of `t` is the integer `D + s·2^n`). When the ranges decide the
    /// comparison, the result is a constant.
    pub(super) fn less_than(&mut self, a: Operand, b: Operand, domain: Option<Interval>) -> Result<Halo2BaseCell, ProvingError> {
        if let (Operand::Cell(x), Operand::Constant(0), Some(Interval::SIGNED64)) = (a, b, domain) {
            if let Some(sign) = self.sign_bits.get(&x.id) {
                return Ok(*sign);
            }
            if let Some(sign) = self.sign_from_window(&x) {
                return Ok(sign);
            }
            if self.proven_range(&x).is_none_or(|r| !r.within(&Interval::SIGNED64)) && self.planned_bound(&x).is_none() {
                return self.sign_test(&x);
            }
        }
        if let (Operand::Cell(x), Operand::Cell(y), Some(Interval::SIGNED64)) = (a, b, domain) {
            if let Some(reverse) = self.orders.get(&(y.id, x.id)).copied() {
                return Ok(self.trichotomy(&x, &y, &reverse));
            }
        }
        if let Some(domain) = domain {
            self.ensure_operand_within(&a, domain)?;
            self.ensure_operand_within(&b, domain)?;
        }
        match (a, b) {
            (Operand::Cell(x), Operand::Constant(c)) => {
                if let Some(tv) = self.two_valued_of(&x) {
                    return self.decide_two_valued(tv, c, false);
                }
            }
            (Operand::Constant(c), Operand::Cell(x)) => {
                if let Some(tv) = self.two_valued_of(&x) {
                    return self.decide_two_valued(tv, c, true);
                }
            }
            _ => {}
        }
        let (ra, rb) = match (self.operand_range(&a), self.operand_range(&b)) {
            (Some(ra), Some(rb)) => (ra, rb),
            _ => return Err(ProvingError::synthesis("comparison operand without a proven range")),
        };
        let d = ra.sub(&rb).ok_or_else(|| ProvingError::synthesis("comparison range overflow"))?;
        if d.hi < 0 {
            return self.constant_bool(true);
        }
        if d.lo >= 0 {
            return self.constant_bool(false);
        }
        let n = self.lookup_aligned(bits_above(d.hi.max(-d.lo)));
        let below = kernel::fp_to_i128(a.value()).zip(kernel::fp_to_i128(b.value())).is_some_and(|(x, y)| x < y);
        let s = self.hint_bit(below);
        let power = kernel::i128_to_fp(1i128 << n);
        let diff = match (a, b) {
            (_, Operand::Constant(0)) => a.quantum(),
            (Operand::Cell(x), Operand::Cell(y)) => self.difference(&x, &y).quantum(),
            _ => self.gate_cell(a.value() - b.value(), |gate, ctx| gate.sub(ctx, a.quantum(), b.quantum())).quantum(),
        };
        let diff_value = a.value() - b.value();
        let t = self.gate_cell(diff_value + s.value * power, |gate, ctx| {
            gate.mul_add(ctx, s.quantum(), QuantumCell::Constant(power), diff)
        });
        self.range_check(&t, n);
        self.record_comparison(&s, a, b);
        if let (Operand::Cell(x), Operand::Cell(y), Some(Interval::SIGNED64)) = (a, b, domain) {
            self.orders.insert((x.id, y.id), s);
        }
        Ok(s)
    }

    /// `[x < y]` from the already pinned `[y < x]` (`reverse`): `1 - [y < x]
    /// - [x == y]`, an equality test and two gates instead of a range check
    /// of the gap.
    ///
    /// Soundness: `reverse` was computed with both operands constrained into
    /// the signed 64-bit window, where exactly one of `x < y`, `x == y`,
    /// `y < x` holds; `[x == y]` is pinned by its zero test. So the result is
    /// 0 or 1, and 1 exactly when `x < y`.
    fn trichotomy(&mut self, x: &Halo2BaseCell, y: &Halo2BaseCell, reverse: &Halo2BaseCell) -> Halo2BaseCell {
        let difference = self.difference(y, x);
        let equal = self.is_zero_bit(&difference);
        let either = self.gate_cell(reverse.value + equal.value, |gate, ctx| gate.add(ctx, reverse.quantum(), equal.quantum()));
        let below = self.gate_cell(Fp::one() - either.value, |gate, ctx| gate.not(ctx, either.quantum()));
        self.mark_bit(&below);
        self.orders.insert((x.id, y.id), below);
        below
    }

    /// `a < b` (`strict`) or `a <= b` as an assertion, for a comparison
    /// whose only use is `Assert` (see `plan.rs`): the operands are
    /// constrained into the signed 64-bit window, then the gap
    /// `g = b - a - [strict]` is range-checked to `[0, 2^n)`, `2^n` above the
    /// largest gap the operands' proven ranges allow. Returns the constant
    /// `true`, which the `Assert` then pins at no cost.
    ///
    /// Soundness: both operands lie in the window, so the gap is an integer
    /// in `(-2^65, 2^65)`; a negative one is the field element `p - |g|`,
    /// far above `2^n`, and fails the check. The relation therefore holds for
    /// every accepted witness, which is what `Assert(a < b)` states. A gap the
    /// ranges already prove non-negative needs no check.
    pub(super) fn assert_relation(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell, strict: bool) -> Result<Halo2BaseCell, ProvingError> {
        let (a, b) = (self.operand(a), self.operand(b));
        self.ensure_operand_within(&a, Interval::SIGNED64)?;
        self.ensure_operand_within(&b, Interval::SIGNED64)?;
        let (ra, rb) = match (self.operand_range(&a), self.operand_range(&b)) {
            (Some(ra), Some(rb)) => (ra, rb),
            _ => return Err(ProvingError::synthesis("comparison operand without a proven range")),
        };
        let slack = i128::from(strict);
        let gap_range = rb.sub(&ra).and_then(|g| g.offset(-slack)).ok_or_else(|| ProvingError::synthesis("comparison range overflow"))?;
        let gap_value = b.value() - a.value() - kernel::i128_to_fp(slack);
        if kernel::fp_to_i128(gap_value).is_none_or(|g| g < 0) {
            self.note_unsat(|| "Witness does not satisfy circuit constraints: an asserted comparison is false".to_string());
        }
        if gap_range.lo < 0 {
            let gap = match (a, b, slack) {
                (Operand::Cell(x), Operand::Constant(0), 0) => self.negate(&x),
                (Operand::Constant(0), Operand::Cell(y), 0) => y,
                (Operand::Cell(x), Operand::Cell(y), 0) => self.difference(&y, &x),
                _ => self.gate_cell(gap_value, |gate, ctx| {
                    let diff = gate.sub(ctx, b.quantum(), a.quantum());
                    gate.sub(ctx, diff, QuantumCell::Constant(kernel::i128_to_fp(slack)))
                }),
            };
            let width = self.lookup_aligned(bits_above(gap_range.hi).max(1));
            self.range_check(&gap, width);
        }
        // a < b (or <=) now holds: narrow a cell compared with a constant.
        match (a, b) {
            (Operand::Cell(x), Operand::Constant(c)) => {
                if let Some(r) = self.proven_range(&x).and_then(|r| Interval::new(r.lo, r.hi.min(c - slack))) {
                    self.prove_range(&x, r);
                }
            }
            (Operand::Constant(c), Operand::Cell(y)) => {
                if let Some(r) = self.proven_range(&y).and_then(|r| Interval::new(r.lo.max(c + slack), r.hi)) {
                    self.prove_range(&y, r);
                }
            }
            _ => {}
        }
        self.constant_bool(true)
    }

    /// `[x < 0]` for an `x` not yet known to lie in the signed 64-bit
    /// window, proving the window at the same time: the hint bit `s` and one
    /// 63-bit range check of `x + s·2^63`. For `s = 0` that is
    /// `0 <= x < 2^63`, for `s = 1` it is `-2^63 <= x < 0`, so `s` is the
    /// only admissible value and `x` lies in the window either way. One check
    /// instead of the window check plus the comparison's.
    fn sign_test(&mut self, x: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let s = self.hint_bit(kernel::fp_is_negative(x.value));
        let power = kernel::i128_to_fp(1i128 << 63);
        let shifted = self.gate_cell(x.value + s.value * power, |gate, ctx| {
            gate.mul_add(ctx, s.quantum(), QuantumCell::Constant(power), x.quantum())
        });
        self.range_check(&shifted, 63);
        self.prove_range(x, Interval::SIGNED64);
        self.record_sign(x, &s);
        Ok(s)
    }

    /// The 64-bit range check of `shifted = x + 2^63` (halo2-base's own
    /// `range_check`, the same cells), keeping its top limb. halo2-base
    /// decomposes a value of `n = ceil(64 / L)` limbs (`L` the lookup width)
    /// by one inner product whose limb cells it looks up; the top limb `l`
    /// sits at a fixed offset of that inner product (halo2-base 004e12c,
    /// `RangeChip::_range_check`, which returns it but is private) and is
    /// checked to `w = 64 - L(n - 1)` bits. `None` for a single-limb
    /// check (`L >= 64`), where no limb is kept.
    fn window_range_check(&mut self, shifted: &Halo2BaseCell) -> Option<WindowCheck> {
        let lookup = self.range_chip().lookup_bits();
        let limbs = 64usize.div_ceil(lookup);
        let assigned = self.assigned(shifted);
        let offset = self.ctx().advice.len() as isize;
        let range = self.range_chip();
        range.range_check(self.ctx(), assigned, 64);
        self.prove_range(shifted, Interval::UNSIGNED64);
        if limbs < 2 {
            return None;
        }
        let top_assigned = self.ctx().get(offset + 1 + 3 * (limbs as isize - 2));
        let top_bits = 64 - lookup * (limbs - 1);
        let expected = kernel::fp_to_i128(shifted.value).map(|v| (v as u128 >> (lookup * (limbs - 1))) as u64);
        let value = *top_assigned.value();
        debug_assert!(expected.is_none_or(|e| value == Fp::from(e)), "window check: top limb not where expected");
        let top_limb = self.wrap(top_assigned, value);
        Some(WindowCheck { shifted: *shifted, top_limb, top_bits })
    }

    /// `[x < 0]` for an `x` whose int64 window check kept its top limb
    /// (`WindowCheck`), without a comparison of its own.
    ///
    /// With `x + 2^63 = sum_i l_i 2^(L i)` and the top limb `l` in
    /// `[0, 2^w)` (the window check's lookups), `x >= 0` exactly when
    /// `x + 2^63 >= 2^63`, i.e. when `l >= 2^(w-1)`. The hint `b = [x >= 0]`
    /// is a bit and `t = l - 2^(w-1) b` is range-checked to `w - 1` bits:
    /// `b = 1` needs `l` in `[2^(w-1), 2^w)`, `b = 0` needs `l` in
    /// `[0, 2^(w-1))`, so exactly one `b` passes and it is the sign. For
    /// `w = 1` the limb is that bit itself. The result `1 - b` is recorded
    /// as `x`'s sign bit. About 20 cells against about 50 for a 64-bit
    /// comparison with 0 (lookup width 8).
    pub(super) fn sign_from_window(&mut self, x: &Halo2BaseCell) -> Option<Halo2BaseCell> {
        let window = self.windows.get(&x.id).copied()?;
        let nonneg = if window.top_bits == 1 {
            self.mark_bit(&window.top_limb);
            window.top_limb
        } else {
            let half = Fp::from(1u64 << (window.top_bits - 1));
            let b = self.hint_bit(!kernel::fp_is_negative(x.value));
            let top = window.top_limb;
            let t = self.gate_cell(top.value - half * b.value, |gate, ctx| {
                gate.sub_mul(ctx, top.quantum(), b.quantum(), QuantumCell::Constant(half))
            });
            self.range_check(&t, window.top_bits - 1);
            b
        };
        let sign = self.gate_cell(Fp::one() - nonneg.value, |gate, ctx| gate.not(ctx, nonneg.quantum()));
        self.mark_bit(&sign);
        self.record_sign(x, &sign);
        Some(sign)
    }

    /// Record `s` as the pinned bit `[x < 0]` of an `x` proven to lie in the
    /// signed 64-bit window: comparisons of `x` with 0 and `sign_and_abs`
    /// reuse it.
    pub(super) fn record_sign(&mut self, x: &Halo2BaseCell, s: &Halo2BaseCell) {
        self.sign_bits.insert(x.id, *s);
        self.record_comparison(s, Operand::Cell(*x), Operand::Constant(0));
    }

    /// Remember that the bit `s` is `[x < c]` or `[c < x]` for a cell `x`
    /// and a constant `c`, so that asserting it (or its negation) refines
    /// `x`'s proven range (see `refine_by_assertion`).
    fn record_comparison(&mut self, s: &Halo2BaseCell, a: Operand, b: Operand) {
        let fact = match (a, b) {
            (Operand::Cell(x), Operand::Constant(c)) => Comparison { x, bound: c, x_below: true },
            (Operand::Constant(c), Operand::Cell(x)) => Comparison { x, bound: c, x_below: false },
            _ => return,
        };
        self.comparisons.insert(s.id, fact);
    }

    /// After `assert_true(bit)`: if the bit is a recorded comparison with a
    /// constant, or the negation of one, the constraints now force the
    /// compared cell to one side of the constant; record that range. The
    /// comparison's own checks pin the bit to the comparison's truth value,
    /// so the asserted bit being 1 implies the relation.
    pub(super) fn refine_by_assertion(&mut self, bit: &Halo2BaseCell) {
        let (fact, holds) = match self.comparisons.get(&bit.id) {
            Some(fact) => (*fact, true),
            None => match self.negations.get(&bit.id).and_then(|s| self.comparisons.get(s)) {
                Some(fact) => (*fact, false),
                None => return,
            },
        };
        let Some(known) = self.proven_range(&fact.x) else { return };
        // x < c holds: x <= c - 1; fails: x >= c. c < x holds: x >= c + 1;
        // fails: x <= c.
        let refined = match (fact.x_below, holds) {
            (true, true) => Interval::new(known.lo, known.hi.min(fact.bound - 1)),
            (true, false) => Interval::new(known.lo.max(fact.bound), known.hi),
            (false, true) => Interval::new(known.lo.max(fact.bound + 1), known.hi),
            (false, false) => Interval::new(known.lo, known.hi.min(fact.bound)),
        };
        if let Some(range) = refined {
            self.prove_range(&fact.x, range);
        }
    }

    /// Assert `a < b` for operands constrained into `[0, 2^64)` (the IR's
    /// unsigned bound assertion), and record `a`'s new range.
    ///
    /// An `a` without a proven range is first range-checked to the narrowest
    /// lookup-aligned width that holds every admissible value (`b`'s upper
    /// bound minus one); a constant power-of-two bound `2^k` is exactly the
    /// `k`-bit check and needs nothing else. Then the gap `b - a - 1` is
    /// range-checked to a width above its largest possible value: with both
    /// operands' ranges far below `p / 2`, the gap is a negative integer, a
    /// field element above `p - 2^120`, exactly when `a >= b`.
    pub(super) fn assert_less(&mut self, a: &Halo2BaseCell, b: Operand) -> Result<(), ProvingError> {
        self.ensure_operand_within(&b, Interval::UNSIGNED64)?;
        let rb = self.operand_range(&b).ok_or_else(|| ProvingError::synthesis("bound without a proven range"))?;
        if self.proven_range(a).is_none_or(|r| !r.within(&Interval::UNSIGNED64)) {
            if let Operand::Constant(c) = b {
                if c > 0 && (c as u128).is_power_of_two() {
                    self.range_check(a, c.trailing_zeros() as usize);
                    return Ok(());
                }
            }
            // Every admissible `a` lies in `[0, rb.hi - 1]` (with no admissible
            // `a` when `rb.hi <= 0`: the gap check below then fails).
            let width = self.lookup_aligned(bits_above(rb.hi - 1).max(1));
            self.range_check(a, width.min(64));
        }
        let ra = self.proven_range(a).expect("range-checked above");
        let gap_range = rb.sub(&ra).and_then(|g| g.offset(-1)).ok_or_else(|| ProvingError::synthesis("bound range overflow"))?;
        if gap_range.lo < 0 {
            let gap_value = b.value() - a.value - Fp::one();
            let gap = match b {
                Operand::Constant(c) => {
                    let limit = kernel::i128_to_fp(c - 1);
                    self.gate_cell(gap_value, |gate, ctx| gate.sub(ctx, QuantumCell::Constant(limit), a.quantum()))
                }
                Operand::Cell(bc) => self.gate_cell(gap_value, |gate, ctx| {
                    let diff = gate.sub(ctx, bc.quantum(), a.quantum());
                    gate.sub(ctx, diff, QuantumCell::Constant(Fp::one()))
                }),
            };
            let width = self.lookup_aligned(bits_above(gap_range.hi).max(1));
            self.range_check(&gap, width);
        }
        if let Some(r) = Interval::new(ra.lo.max(0), ra.hi.min(rb.hi - 1)) {
            self.prove_range(a, r);
        }
        Ok(())
    }
}
