//! Bitwise `&`, `|`, `^` and `~` on int64 values, with NumPy's two's
//! complement semantics (the mock computes them natively on `i64`).
//!
//! Everything reduces to `&` through two integer identities of two's
//! complement: `a | b = a + b - (a & b)` and `a ^ b = a + b - 2·(a & b)`,
//! and `~a = -a - 1`. `a & b` is `Σ a_i·b_i·2^i` over the operands' bits,
//! with weight `-2^63` for bit 63. Only as many low bits are decomposed as
//! the result can have: a constant mask `c >= 0` needs `bit_length(c)` bits
//! of the other operand, an operand with a proven range `[0, 2^w)` limits
//! the result to `w` bits, otherwise all 64.

use halo2_base::gates::GateInstructions;
use halo2_base::QuantumCell;

use super::ranges::{Interval, TwoValued};
use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::prove::traits::Synthesizer;

/// Bit width of int64 values.
const WORD: usize = 64;

impl Halo2BaseSynthesizer<'_> {
    /// The low `width` two's-complement bits of `x`, least significant first
    /// (`width <= 64`; for 64, bit 63 is the sign bit). Cached per cell; a
    /// cached wider decomposition is reused.
    ///
    /// Soundness: the bits are hints, each asserted to be a bit, recombined
    /// by one inner product.
    /// * `x` proven in `[0, 2^w)`: `x = Σ_{i<w} b_i·2^i` (bits above `w` are
    ///   the constant 0). The sum is below `2^w ≪ p`, so the binary
    ///   representation is unique.
    /// * `width = 64`: `x + 2^63 = Σ_{i<64} b'_i·2^i`, which also proves `x` in
    ///   the signed 64-bit window; `x`'s bits are `b'_i` for `i < 63` and
    ///   `1 - b'_63` for the sign.
    /// * otherwise: a hint `h = floor(x / 2^width)`, the gate
    ///   `low = x - h·2^width`, `low = Σ_{i<width} b_i·2^i`, and
    ///   `h + 2^(63-width)` range-checked to `64 - width` bits. Then
    ///   `|low + h·2^width| < 2^63 + 2^width ≪ p/2`, so `x = h·2^width + low`
    ///   over the integers with `0 <= low < 2^width`: `low = x mod 2^width`,
    ///   whose bits are `x`'s low two's-complement bits. `x` outside the
    ///   window has no witness.
    pub(super) fn low_bits(&mut self, x: &Halo2BaseCell, width: usize) -> Result<Vec<Halo2BaseCell>, ProvingError> {
        let width = width.min(WORD);
        if let Some(bits) = self.bit_cache.get(&x.id).filter(|bits| bits.len() >= width) {
            return Ok(bits[..width].to_vec());
        }
        if let Some(v) = self.is_const_cell(x).then(|| kernel::fp_to_i64(x.value)) {
            return (0..width).map(|i| self.constant_bool((v >> i) & 1 == 1)).collect();
        }
        let natural = self.proven_range(x).filter(|r| r.lo >= 0 && r.hi < 1i128 << 63).map(|r| bit_length(r.hi));
        let bits = match natural {
            Some(w) if w <= width => {
                let mut bits = self.hinted_bits(x, w)?;
                for _ in w..width {
                    bits.push(self.constant_bool(false)?);
                }
                bits
            }
            _ if width == WORD => {
                let offset = Fp::from(1u64 << 63);
                let shifted = self.gate_cell(x.value + offset, |gate, ctx| {
                    gate.add(ctx, x.assigned, QuantumCell::Constant(offset))
                });
                let mut bits = self.hinted_bits(&shifted, WORD)?;
                bits[WORD - 1] = self.logical_not(&bits[WORD - 1])?;
                self.prove_range(x, Interval::SIGNED64);
                bits
            }
            _ => {
                let value = kernel::fp_to_i128(x.value).unwrap_or(0);
                let power = kernel::i128_to_fp(1i128 << width);
                let high = self.hint(kernel::i128_to_fp(value >> width));
                let low = self.gate_cell(x.value - high.value * power, |gate, ctx| {
                    gate.sub_mul(ctx, x.assigned, high.assigned, QuantumCell::Constant(power))
                });
                let window = kernel::i128_to_fp(1i128 << (63 - width));
                let shifted = self.gate_cell(high.value + window, |gate, ctx| {
                    gate.add(ctx, high.assigned, QuantumCell::Constant(window))
                });
                self.range_check(&shifted, WORD - width);
                self.prove_range(x, Interval::SIGNED64);
                self.hinted_bits(&low, width)?
            }
        };
        self.bit_cache.insert(x.id, bits.clone());
        Ok(bits)
    }

    /// Bits `b_0..b_{w-1}` of `x`, pinned by `x = Σ b_i·2^i` and a bit check
    /// each (halo2-base's `num_to_bits` layout, with the bits as hints).
    fn hinted_bits(&mut self, x: &Halo2BaseCell, width: usize) -> Result<Vec<Halo2BaseCell>, ProvingError> {
        let value = kernel::fp_to_i128(x.value).unwrap_or(0) as u128;
        let bit_values: Vec<Fp> = (0..width)
            .map(|i| if (value >> i) & 1 == 1 { Fp::one() } else { Fp::zero() })
            .map(|v| self.hint_value(v))
            .collect();
        let powers: Vec<Fp> = (0..width).map(|i| kernel::i128_to_fp(1i128 << i)).collect();
        let row = self.builder.main(0).advice.len();
        let sum = self.gate.inner_product(
            self.builder.main(0),
            bit_values.iter().map(|b| QuantumCell::Witness(*b)),
            powers.iter().map(|p| QuantumCell::Constant(*p)),
        );
        let ctx = self.builder.main(0);
        ctx.constrain_equal(&x.assigned, &sum);
        // inner_product lays out | b_0 | (c_1 | b_1 | acc_1) ... : b_0 at
        // `row`, b_i at `row + 1 + 3(i - 1)`.
        let assigned: Vec<_> = (0..width)
            .map(|i| ctx.get((if i == 0 { row } else { row + 1 + 3 * (i - 1) }) as isize))
            .collect();
        let mut bits = Vec::with_capacity(width);
        for (cell, value) in assigned.into_iter().zip(bit_values) {
            let bit = self.wrap(cell, value);
            self.gate.assert_bit(self.builder.main(0), bit.assigned);
            self.mark_bit(&bit);
            bits.push(bit);
        }
        Ok(bits)
    }

    /// `a & b`.
    pub(super) fn bit_and(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        match (self.int_constant(a), self.int_constant(b)) {
            (Some(x), Some(y)) => self.constant_int(x & y),
            (Some(c), None) => self.and_constant(b, c),
            (None, Some(c)) => self.and_constant(a, c),
            (None, None) => self.and_cells(a, b),
        }
    }

    /// `a | b = a + b - (a & b)`.
    pub(super) fn bit_or(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let both = self.bit_and(a, b)?;
        let sum = self.add_i(a, b)?;
        self.sub_i(&sum, &both)
    }

    /// `a ^ b = a + b - 2·(a & b)`.
    pub(super) fn bit_xor(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let both = self.bit_and(a, b)?;
        let twice = self.add_i(&both, &both)?;
        let sum = self.add_i(a, b)?;
        self.sub_i(&sum, &twice)
    }

    /// `~a = -a - 1`.
    pub(super) fn bit_not(&mut self, a: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let minus_one = self.constant_int(-1)?;
        self.sub_i(&minus_one, a)
    }

    fn int_constant(&self, x: &Halo2BaseCell) -> Option<i64> {
        self.is_const_cell(x).then(|| kernel::fp_to_i64(x.value)).filter(|v| kernel::i64_to_fp(*v) == x.value)
    }

    /// `x & c` for a constant mask: the selected low bits of `x` for
    /// `c >= 0`, and `x - (x & ~c)` for `c < 0` (the two masks partition
    /// the bits).
    fn and_constant(&mut self, x: &Halo2BaseCell, c: i64) -> Result<Halo2BaseCell, ProvingError> {
        if c < 0 {
            let rest = self.and_constant(x, !c)?;
            return self.sub_i(x, &rest);
        }
        let width = bit_length(c as i128);
        if width == 0 {
            return self.constant_int(0);
        }
        let bits = self.low_bits(x, width)?;
        let selected: Vec<(Halo2BaseCell, i128)> =
            (0..width).filter(|i| (c >> i) & 1 == 1).map(|i| (bits[i], 1i128 << i)).collect();
        let out = self.weighted_sum(&selected)?;
        self.prove_range(&out, Interval { lo: 0, hi: c as i128 });
        if let [(bit, weight)] = selected[..] {
            self.set_two_valued(&out, TwoValued { bit, if_zero: 0, if_one: weight });
        }
        Ok(out)
    }

    /// `a & b` of two cells: `Σ a_i·b_i·2^i` over the bits the result can
    /// have (bit 63 weighted `-2^63`).
    fn and_cells(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let natural = |r: Option<Interval>| r.filter(|r| r.lo >= 0 && r.hi < 1i128 << 63).map(|r| bit_length(r.hi));
        let width = match (natural(self.proven_range(a)), natural(self.proven_range(b))) {
            (Some(x), Some(y)) => x.min(y),
            (Some(w), None) | (None, Some(w)) => w,
            (None, None) => WORD,
        };
        if width == 0 {
            return self.constant_int(0);
        }
        let a_bits = self.low_bits(a, width)?;
        let b_bits = self.low_bits(b, width)?;
        let mut terms = Vec::with_capacity(width);
        for (i, (x, y)) in a_bits.iter().zip(&b_bits).enumerate() {
            let both = self.logical_and(x, y)?;
            let weight = if i == WORD - 1 { -(1i128 << 63) } else { 1i128 << i };
            terms.push((both, weight));
        }
        let out = self.weighted_sum(&terms)?;
        let range = if width == WORD { Interval::SIGNED64 } else { Interval { lo: 0, hi: (1i128 << width) - 1 } };
        self.prove_range(&out, range);
        Ok(out)
    }

    /// `Σ weight·cell` for constant weights (one inner product; the cell
    /// itself for a single unit weight).
    fn weighted_sum(&mut self, terms: &[(Halo2BaseCell, i128)]) -> Result<Halo2BaseCell, ProvingError> {
        match terms {
            [] => self.constant_int(0),
            [(cell, 1)] => Ok(*cell),
            _ => {
                let value: Fp = terms.iter().map(|(c, w)| c.value * kernel::i128_to_fp(*w)).sum();
                Ok(self.gate_cell(value, |gate, ctx| {
                    gate.inner_product(
                        ctx,
                        terms.iter().map(|(c, _)| QuantumCell::Existing(c.assigned)),
                        terms.iter().map(|(_, w)| QuantumCell::Constant(kernel::i128_to_fp(*w))),
                    )
                }))
            }
        }
    }
}

/// The number of bits of `v >= 0` (0 for 0).
fn bit_length(v: i128) -> usize {
    (128 - v.max(0).leading_zeros()) as usize
}
