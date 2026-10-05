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
            Operand::Cell(c) => QuantumCell::Existing(c.assigned),
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
    pub(super) fn prove_range(&mut self, x: &Halo2BaseCell, range: Interval) {
        let range = match self.proven.get(&x.id) {
            Some(known) => known.intersect(&range).unwrap_or(range),
            None => range,
        };
        self.proven.insert(x.id, range);
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

    fn operand_range(&self, x: &Operand) -> Option<Interval> {
        match x {
            Operand::Cell(c) => self.proven_range(c),
            Operand::Constant(v) => Interval::point(*v),
        }
    }

    /// Constrain `0 <= x < 2^bits` (and record it).
    pub(super) fn range_check(&mut self, x: &Halo2BaseCell, bits: usize) {
        let range = self.range_chip();
        range.range_check(self.ctx(), x.assigned, bits);
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
        if domain == Interval::UNSIGNED64 {
            self.range_check(x, 64);
        } else if domain == Interval::SIGNED64 {
            let shift = Fp::from(1u64 << 63);
            let shifted = self.gate_cell(x.value + shift, |gate, ctx| {
                gate.add(ctx, x.assigned, QuantumCell::Constant(shift))
            });
            self.range_check(&shifted, 64);
        } else {
            return Err(ProvingError::synthesis("unsupported comparison domain"));
        }
        self.prove_range(x, domain);
        Ok(())
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
        if let Some(domain) = domain {
            self.ensure_operand_within(&a, domain)?;
            self.ensure_operand_within(&b, domain)?;
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
        let diff = match b {
            Operand::Constant(0) => a.quantum(),
            _ => QuantumCell::Existing(self.gate_cell(a.value() - b.value(), |gate, ctx| gate.sub(ctx, a.quantum(), b.quantum())).assigned),
        };
        let diff_value = a.value() - b.value();
        let t = self.gate_cell(diff_value + s.value * power, |gate, ctx| {
            gate.mul_add(ctx, s.assigned, QuantumCell::Constant(power), diff)
        });
        self.range_check(&t, n);
        Ok(s)
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
                    self.gate_cell(gap_value, |gate, ctx| gate.sub(ctx, QuantumCell::Constant(limit), a.assigned))
                }
                Operand::Cell(bc) => self.gate_cell(gap_value, |gate, ctx| {
                    let diff = gate.sub(ctx, bc.assigned, a.assigned);
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
