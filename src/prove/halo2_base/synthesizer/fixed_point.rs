//! Fixed-point and integer arithmetic gadgets: constrained division and
//! rescaling, sign and magnitude, square roots and polynomial evaluation.
//! Every prover hint here is pinned by the constraints stated next to it.

use halo2_base::gates::GateInstructions;
use halo2_base::QuantumCell;

use super::ranges::{Interval, Operand, TwoValued};
use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel::{self, Rounding};
use crate::prove::traits::Synthesizer;

/// The divisor of a constrained division, classified by what the circuit
/// already knows about it (see `div_rem`).
#[derive(Clone, Copy)]
pub(super) enum Divisor {
    /// A fixed-column constant in the signed 64-bit window.
    Constant(i128),
    /// A cell the caller knows to be positive, so the remainder needs no
    /// sign normalisation. The remainder bound re-proves `1 <= b <= 2^65`,
    /// so a wrong claim makes the circuit unsatisfiable, not unsound.
    Positive(Halo2BaseCell),
    /// A cell about which nothing is known except that it is non-zero
    /// (`div_rem` maps a zero divisor away before using this).
    NonZero(Halo2BaseCell),
}


/// What the circuit knows about a Q-encoded value used as a factor.
#[derive(Clone, Copy)]
pub(super) enum Factor {
    /// `n · 2^P` for the integer cell `n` (a float cast).
    Integer(Halo2BaseCell),
    /// The constant `m · 2^s`, with `s <= P` and `m` odd (or `m = 0`).
    Constant { m: i128, s: u32 },
    Unknown,
}

impl Halo2BaseSynthesizer<'_> {
    /// `(s, s ? -x : x)` for a hinted sign bit `s = [x < 0]`. The sign is
    /// NOT pinned here: the caller must constrain the magnitude into a
    /// window far below p, which rules out the wrong sign (it would make
    /// the magnitude the field negation of a small number, about p).
    pub(super) fn hinted_sign_and_magnitude(
        &mut self,
        x: &Halo2BaseCell,
    ) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
        let sign = self.hint_bit(kernel::fp_is_negative(x.value));
        let neg_x = self.gate_cell(-x.value, |gate, ctx| gate.neg(ctx, x.quantum()));
        let magnitude = self.select_bit(&sign, &neg_x, x)?;
        Ok((sign, magnitude))
    }


    /// `(s, m)` with `s = [x < 0]` and `m = |x|`; an `x` outside the signed
    /// 64-bit window has no witness.
    ///
    /// Soundness: `m = s ? -x : x` with a hinted sign bit, and the range check
    /// `m - s ∈ [0, 2^63)`. For `s = 0` that is `0 <= x < 2^63`; for `s = 1`
    /// it is `-x - 1 ∈ [0, 2^63)`, i.e. `-2^63 <= x <= -1`. The wrong sign
    /// makes `m - s` the field negation of a small number (about p), so `s`
    /// and `m` are functions of `x`. The sign is then `x`'s recorded sign bit
    /// (comparisons of `x` with 0 reuse it); a sign bit recorded before (by a
    /// sign test, which pins it the same way) needs only the select.
    pub(super) fn sign_and_abs(&mut self, x: &Halo2BaseCell) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
        if let Some(pair) = self.sign_abs_cache.get(&x.id) {
            return Ok(*pair);
        }
        let magnitude_range = Interval { lo: 0, hi: 1 << 63 };
        let pair = if self.is_const_cell(x) {
            let negative = kernel::fp_is_negative(x.value);
            let sign = self.constant_bool(negative)?;
            let magnitude = self.const_cell(if negative { -x.value } else { x.value });
            (sign, magnitude)
        } else if let Some(sign) = self.sign_bits.get(&x.id).copied() {
            let neg_x = self.negate(x);
            let magnitude = self.select_bit(&sign, &neg_x, x)?;
            self.prove_range(&magnitude, magnitude_range);
            (sign, magnitude)
        } else {
            let (sign, magnitude) = self.hinted_sign_and_magnitude(x)?;
            let shifted = self.sub_i(&magnitude, &sign)?;
            self.range_check(&shifted, 63);
            self.prove_range(x, Interval::SIGNED64);
            self.prove_range(&magnitude, magnitude_range);
            self.record_sign(x, &sign);
            (sign, magnitude)
        };
        self.sign_abs_cache.insert(x.id, pair);
        Ok(pair)
    }


    /// `|x|` alone (no sign bit for later use).
    ///
    /// For an `x` already proven inside the signed 64-bit window `[lo, hi]`
    /// (an input, a checked result): a hinted bit `s`, `m = x·(1 - 2s)` (two
    /// gates) and `m` range-checked to the lookup-aligned width `w` of
    /// `max(|lo|, |hi|)`. The right `s` gives `m = |x| < 2^w`; for `x != 0`
    /// the wrong one gives `m = p - |x|`, far above `2^w`, so `m` is a
    /// function of `x`. At `x = 0`, `m = 0` for either `s`: `s` is not
    /// pinned there, which is why it is neither returned nor recorded as
    /// `x`'s sign (a sign test reusing it would be unsound). Otherwise (the
    /// window not yet proven, or a sign already known) `sign_and_abs`.
    pub(super) fn magnitude(&mut self, x: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let known_sign = self.sign_abs_cache.contains_key(&x.id) || self.sign_bits.contains_key(&x.id);
        let window = self.proven_range(x).filter(|r| r.within(&Interval::SIGNED64));
        let (Some(range), false, false) = (window, known_sign, self.is_const_cell(x)) else {
            let magnitude = self.sign_and_abs(x)?.1;
            return Ok(self.below_two_pow_63(magnitude));
        };
        // `|i64::MIN| = 2^63` overflows int64 (NumPy wraps it to i64::MIN):
        // like every int64 overflow here it has no proof, so a 63-bit check
        // when `x` may be `i64::MIN`, keeping `|x|` inside the window.
        let max_abs = range.lo.unsigned_abs().max(range.hi.unsigned_abs()).min((1u128 << 63) - 1);
        let bits = (128 - max_abs.leading_zeros()) as usize;
        let width = if bits >= 63 { 63 } else { self.lookup_aligned(bits.max(1)) };
        let sign = self.hint_bit(kernel::fp_is_negative(x.value));
        let two = Fp::from(2u64);
        let factor = self.gate_cell(Fp::one() - two * sign.value, |gate, ctx| {
            gate.mul_add(ctx, sign.quantum(), QuantumCell::Constant(-two), QuantumCell::Constant(Fp::one()))
        });
        let magnitude = self.gate_cell(x.value * factor.value, |gate, ctx| gate.mul(ctx, x.quantum(), factor.quantum()));
        self.range_check(&magnitude, width);
        self.prove_range(&magnitude, Interval { lo: 0, hi: max_abs as i128 });
        Ok(magnitude)
    }

    /// `|x|` from `sign_and_abs`, which allows `2^63` (`x = i64::MIN`),
    /// restricted to `[0, 2^63)`: `|i64::MIN|` overflows int64 and, like
    /// every int64 overflow, has no proof. One 63-bit range check unless the
    /// magnitude's proven range already excludes `2^63`.
    fn below_two_pow_63(&mut self, magnitude: Halo2BaseCell) -> Halo2BaseCell {
        let limit = (1i128 << 63) - 1;
        if self.proven_range(&magnitude).is_some_and(|r| r.lo >= 0 && r.hi <= limit) {
            return magnitude;
        }
        self.range_check(&magnitude, 63);
        self.prove_range(&magnitude, Interval { lo: 0, hi: limit });
        magnitude
    }

    /// Constrain `0 <= x < n` for a cell `n`. Both `x` and `n - 1 - x` are
    /// range-checked to 64 bits, so `n - 1`, their sum, lies in `[0, 2^65)`
    /// over the integers: the check also proves `1 <= n <= 2^65`.
    pub(super) fn assert_below_cell(&mut self, x: &Halo2BaseCell, n: &Halo2BaseCell) {
        self.range_check(x, 64);
        let gap = self.gate_cell(n.value - Fp::one() - x.value, |gate, ctx| {
            let n_minus_x = gate.sub(ctx, n.quantum(), x.quantum());
            gate.dec(ctx, n_minus_x)
        });
        self.range_check(&gap, 64);
        self.prove_range(n, Interval { lo: 1, hi: 1 << 65 });
    }


    /// The divisor kind of `b` for [`Self::div_rem_nonzero`]: a constant in
    /// the signed 64-bit window, or `None` for a cell to be handled
    /// dynamically.
    pub(super) fn constant_divisor(&self, b: &Halo2BaseCell) -> Option<i128> {
        if !self.is_const_cell(b) {
            return None;
        }
        kernel::fp_to_i128(b.value).filter(|c| c.unsigned_abs() < 1u128 << 63)
    }


    /// Constrained signed division `a = b*q + r` with the remainder sign of
    /// `rounding`; returns `(q, r)`. A zero divisor yields `(0, 0)`, the
    /// shared convention of `kernel::div_rem_or_zero` and the mock backend.
    ///
    /// A dynamic divisor `b` is first mapped to `b' = b + [b == 0]` and the
    /// dividend to `a' = a * [b != 0]`: for `b = 0` this is the division
    /// `0 / 1`, which forces `(0, 0)`; otherwise it is the original one.
    pub(super) fn div_rem(
        &mut self,
        a: &Halo2BaseCell,
        b: &Halo2BaseCell,
        rounding: Rounding,
    ) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
        match self.constant_divisor(b) {
            Some(0) => {
                let zero = self.constant_int(0)?;
                Ok((zero, zero))
            }
            Some(c) => self.div_rem_nonzero(a, Divisor::Constant(c), rounding),
            // A divisor an emitted constraint proves non-zero needs no zero
            // mapping, and one also proven non-negative is positive.
            None if self.nonzero.contains(&b.id) => {
                let divisor = if self.proven_range(b).is_some_and(|r| r.lo >= 0) {
                    Divisor::Positive(*b)
                } else {
                    Divisor::NonZero(*b)
                };
                self.div_rem_nonzero(a, divisor, rounding)
            }
            None => {
                let b_is_zero = self.is_zero_bit(b);
                let b_nonzero = self.add_i(b, &b_is_zero)?;
                let a_masked = self.gate_cell(
                    (Fp::one() - b_is_zero.value) * a.value,
                    |gate, ctx| gate.mul_not(ctx, b_is_zero.quantum(), a.quantum()),
                );
                self.div_rem_nonzero(&a_masked, Divisor::NonZero(b_nonzero), rounding)
            }
        }
    }


    /// Constrained division by a divisor that is non-zero by construction.
    ///
    /// Constraints (A, B, Q, R are the integers the cells encode):
    ///   * `r = a - b*q` (one gate, exact in the field), `q` a hint;
    ///   * `q` bounded by `bound_quotient`: the signed 64-bit window, or the
    ///     narrower range a constant divisor and `a`'s proven range imply;
    ///   * `0 <= r_adj < |B|`, where `r_adj = ±r` strips the convention's
    ///     remainder sign (Euclid: none; Floor: the divisor's; Trunc: the
    ///     dividend's, from `sign_and_abs`). For a constant divisor this is
    ///     [`Self::assert_less`]; for a cell it is
    ///     [`Self::assert_below_cell`] against `|b|`, which also proves
    ///     `1 <= |B| <= 2^65` and with it the hinted sign of `b`.
    ///
    /// Soundness: `|B·Q + R| < 2^65·2^64 + 2^65`, far below p/2, so
    /// `A = B·Q + R` holds over the integers. A second solution would give
    /// `B(Q - Q') = R' - R` with `|R' - R| < |B|` and equal remainder signs,
    /// so `Q = Q'` and `R = R'`: `q` and `r` are functions of `(a, b)`. An
    /// input whose true quotient leaves the window has no witness.
    pub(super) fn div_rem_nonzero(
        &mut self,
        a: &Halo2BaseCell,
        divisor: Divisor,
        rounding: Rounding,
    ) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
        let (b_fp, b_value, b_quantum) = match divisor {
            Divisor::Constant(c) => {
                let fp = kernel::i128_to_fp(c);
                (fp, Some(c), QuantumCell::Constant(fp))
            }
            Divisor::Positive(b) | Divisor::NonZero(b) => {
                (b.value, kernel::fp_to_i128(b.value), b.quantum())
            }
        };
        let q_value = match (kernel::fp_to_i128(a.value), b_value) {
            (Some(av), Some(bv)) => kernel::div_rem_or_zero(av, bv, rounding).0,
            _ => 0,
        };
        let (q, r) = self.quotient_gate(a, b_quantum, b_fp, kernel::i128_to_fp(q_value));
        let implied = match divisor {
            Divisor::Constant(c) => self.proven_range(a).and_then(|range| quotient_range(range, c)),
            _ => None,
        };
        self.bound_quotient(&q, implied)?;

        // A dynamic divisor's hinted sign and magnitude, pinned by the
        // remainder bound below.
        let dynamic_sign = match divisor {
            Divisor::NonZero(b) => Some(match self.divisor_signs.get(&b.id).copied() {
                Some(pair) => pair,
                None => {
                    let pair = self.hinted_sign_and_magnitude(&b)?;
                    self.divisor_signs.insert(b.id, pair);
                    pair
                }
            }),
            _ => None,
        };
        let r_adj = match rounding {
            Rounding::Euclid => r,
            Rounding::Floor => match (divisor, dynamic_sign) {
                (Divisor::Constant(c), _) if c < 0 => self.negate(&r),
                (_, Some((b_sign, _))) => self.negate_if(&r, &b_sign)?,
                _ => r,
            },
            Rounding::Trunc => {
                let (a_sign, _) = self.sign_and_abs(a)?;
                self.negate_if(&r, &a_sign)?
            }
        };
        match (divisor, dynamic_sign) {
            (Divisor::Constant(c), _) => self.assert_less(&r_adj, Operand::Constant(c.abs()))?,
            (_, Some((_, b_magnitude))) => self.assert_below_cell(&r_adj, &b_magnitude),
            (Divisor::Positive(b), _) => self.assert_below_cell(&r_adj, &b),
            (Divisor::NonZero(_), None) => unreachable!("non-zero divisor without a sign"),
        }
        // With A = B·Q + R over the integers and |R| < |B|, Q = (A - R) / B
        // lies within one of A / B, so in `implied`.
        if let Some(range) = implied {
            self.prove_range(&q, range);
        }
        Ok((q, r))
    }


    /// A hinted quotient `q` of `dividend` by `divisor` and the remainder
    /// `r = dividend - divisor·q`, as one gate `r + divisor·q = dividend`
    /// laid out `[r, divisor, q, dividend]`: `q` and `r` are witnesses of the
    /// gate itself (four cells), not a loaded hint plus a `sub_mul` gate (five).
    /// The caller bounds `q` and `r`; `q` is a hint (see [`Self::hint`]).
    fn quotient_gate(
        &mut self,
        dividend: &Halo2BaseCell,
        divisor: QuantumCell<Fp>,
        divisor_value: Fp,
        q_value: Fp,
    ) -> (Halo2BaseCell, Halo2BaseCell) {
        let q_value = self.hint_value(q_value);
        let r_value = dividend.value - divisor_value * q_value;
        let ctx = self.builder.main(0);
        ctx.assign_region(
            [QuantumCell::Witness(r_value), divisor, QuantumCell::Witness(q_value), dividend.quantum()],
            [0],
        );
        let (r, q) = (ctx.get(-4), ctx.get(-2));
        (self.wrap(q, q_value), self.wrap(r, r_value))
    }

    /// `floor(raw / 2^shift)`, for `shift <= 64`.
    ///
    /// Soundness: the hint `q` and `rem = raw - q·2^shift` (one gate), with
    /// `rem` range-checked to `[0, 2^shift)` and `q` to the signed 64-bit
    /// window, so `q·2^shift + rem` lies in `[-2^(63+shift), 2^(63+shift))`,
    /// far below `p / 2`: it equals `raw` over the integers and `q` is the
    /// unique floor. Without the window check every remainder in
    /// `[0, 2^shift)` would be accepted, each with its own (huge) `q`.
    pub(super) fn rescale_by(&mut self, raw: &Halo2BaseCell, shift: u32) -> Result<Halo2BaseCell, ProvingError> {
        Ok(self.rescale_with_remainder(raw, shift)?.0)
    }

    /// [`Self::rescale_by`], also returning the remainder `raw - q·2^shift`
    /// (proven to lie in `[0, 2^shift)`).
    pub(super) fn rescale_with_remainder(
        &mut self,
        raw: &Halo2BaseCell,
        shift: u32,
    ) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
        let q_value = kernel::fp_to_i128(raw.value).map_or(0, |v| v >> shift);
        let scale = kernel::i128_to_fp(1i128 << shift);
        let (q, rem) = self.quotient_gate(raw, QuantumCell::Constant(scale), scale, kernel::i128_to_fp(q_value));
        self.range_check(&rem, shift as usize);
        // A proven range [lo, hi] of `raw` bounds the floor to
        // [lo >> shift, hi >> shift]; once the identity holds over the
        // integers, `q` lies there (`rem` is non-negative and below 2^shift).
        let implied = self.proven_range(raw).and_then(|r| Interval::new(r.lo >> shift, r.hi >> shift));
        self.bound_quotient(&q, implied)?;
        if let Some(range) = implied {
            self.prove_range(&q, range);
        }
        Ok((q, rem))
    }

    /// What the circuit knows about a Q-encoded value used as a factor.
    pub(super) fn factor_of(&self, x: &Halo2BaseCell) -> Factor {
        let precision_bits = self.params.precision_bits;
        if let Some(n) = self.float_casts.get(&x.id) {
            return Factor::Integer(*n);
        }
        if self.is_const_cell(x) {
            if let Some(v) = kernel::fp_to_i128(x.value).filter(|v| v.unsigned_abs() < 1u128 << 63) {
                let s = if v == 0 { precision_bits } else { v.trailing_zeros().min(precision_bits) };
                return Factor::Constant { m: v >> s, s };
            }
        }
        Factor::Unknown
    }

    /// `x · m` for a constant `m` (one gate, the constant inline).
    fn mul_by_constant(&mut self, x: &Halo2BaseCell, m: i128) -> Result<Halo2BaseCell, ProvingError> {
        let factor = kernel::i128_to_fp(m);
        let out = self.gate_cell(x.value * factor, |gate, ctx| gate.mul(ctx, x.quantum(), QuantumCell::Constant(factor)));
        if let Some(range) = self.proven_range(x).zip(Interval::point(m)).and_then(|(r, c)| r.mul(&c)) {
            self.prove_range(&out, range);
        }
        if let Some(tv) = self.two_valued_of(x) {
            if let (Some(z), Some(o)) = (tv.if_zero.checked_mul(m), tv.if_one.checked_mul(m)) {
                self.set_two_valued(&out, TwoValued { bit: tv.bit, if_zero: z, if_one: o });
            }
        }
        Ok(out)
    }

    /// Fixed-point `a * b = floor(a·b / 2^P)` (the kernel's `fp_mul_rescale`).
    ///
    /// The floor only needs a constraint where the product is not already a
    /// multiple of `2^P`: a factor that is a float cast `n·2^P` of an integer
    /// cell makes the result the exact integer product `a·n`, and a constant
    /// factor `m·2^s` makes it `floor(a·m / 2^(P-s))`, a rescale by fewer bits
    /// (none for an integer-valued constant). Both are identities of the
    /// integers the cells encode, so the value is the kernel's.
    pub(super) fn fixed_mul(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        match (self.factor_of(a), self.factor_of(b)) {
            (Factor::Integer(n), _) => self.mul_i(&n, b),
            (_, Factor::Integer(n)) => self.mul_i(a, &n),
            (Factor::Constant { m, s }, _) => self.scaled_product(b, m, s),
            (_, Factor::Constant { m, s }) => self.scaled_product(a, m, s),
            _ => {
                let raw = self.mul_i(a, b)?;
                self.rescale_by(&raw, precision_bits)
            }
        }
    }

    /// `floor(x·m·2^s / 2^P)` for the constant factor `m·2^s`: the exact
    /// `x·m` when `s = P`, else `floor(x·m / 2^(P-s))` when that remainder
    /// check is cheaper than the full one of `floor(x·(m·2^s) / 2^P)` (an
    /// unaligned width pays for a partial top limb).
    fn scaled_product(&mut self, x: &Halo2BaseCell, m: i128, s: u32) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let shift = precision_bits - s;
        if shift == 0 {
            return self.mul_by_constant(x, m);
        }
        if self.range_check_cost(shift as usize) < self.range_check_cost(precision_bits as usize) {
            let raw = self.mul_by_constant(x, m)?;
            return self.rescale_by(&raw, shift);
        }
        let raw = self.mul_by_constant(x, m << s)?;
        self.rescale_by(&raw, precision_bits)
    }

    /// Fused `floor(Σ a_i·b_i / 2^P)` (the kernel's
    /// `fp_product_sum_rescale`). Products with a float-cast factor `n·2^P`
    /// are multiples of `2^P`, so they leave the floor as the exact `a·n`:
    /// `floor((Σ_exact a·n·2^P + R) / 2^P) = Σ_exact a·n + floor(R / 2^P)`.
    pub(super) fn fixed_product_sum(
        &mut self,
        pairs: &[(Halo2BaseCell, Halo2BaseCell)],
    ) -> Result<Halo2BaseCell, ProvingError> {
        let mut exact: Option<Halo2BaseCell> = None;
        let mut raw: Option<Halo2BaseCell> = None;
        for (a, b) in pairs {
            let integer = match (self.factor_of(a), self.factor_of(b)) {
                (Factor::Integer(n), _) => Some((n, *b)),
                (_, Factor::Integer(n)) => Some((n, *a)),
                _ => None,
            };
            // Each sum is a chain of multiply-add gates sharing their
            // accumulator cells (three cells per term, an inner product).
            match integer {
                Some((n, x)) => {
                    exact = Some(match exact {
                        Some(sum) => self.multiply_add(&sum, &n, &x),
                        None => self.mul_i(&n, &x)?,
                    });
                }
                None => {
                    raw = Some(match raw {
                        Some(acc) => self.multiply_add(&acc, a, b),
                        None => self.product(a, b),
                    });
                }
            }
        }
        let rescaled = match raw {
            Some(raw) => Some(self.rescale_by(&raw, self.params.precision_bits)?),
            None => None,
        };
        match (exact, rescaled) {
            (Some(e), Some(r)) => self.add_i(&e, &r),
            (Some(e), None) => Ok(e),
            (None, Some(r)) => Ok(r),
            (None, None) => self.constant_float(0.0),
        }
    }

    /// Fixed-point `a / b`: `(a·2^P) / b` rounded by `Rounding::Euclid`
    /// (the kernel's `fp_div_prescale`); 0 for a zero divisor.
    ///
    /// With `S = 2^s > 0`, `(A·S) / (B·S)` and `A / B` have the same
    /// Euclidean quotient. So a constant divisor `m·2^s` divides
    /// `a·2^(P-s)` by `m` (a remainder bound of `|m|`, not `|m|·2^s`), and a
    /// float-cast divisor `n·2^P` divides `a` by the integer `n` (with the
    /// same zero mapping: `n = 0` exactly when the divisor is 0).
    pub(super) fn fixed_div(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        match self.factor_of(b) {
            Factor::Integer(n) => Ok(self.div_rem(a, &n, Rounding::Euclid)?.0),
            Factor::Constant { m, .. } if m == 0 => self.constant_float(0.0),
            Factor::Constant { m, s } => {
                let numerator = self.scaled_by_power_of_two(a, precision_bits - s)?;
                Ok(self.div_rem_nonzero(&numerator, Divisor::Constant(m), Rounding::Euclid)?.0)
            }
            Factor::Unknown => {
                let numerator = self.scaled_by_power_of_two(a, precision_bits)?;
                Ok(self.div_rem(&numerator, b, Rounding::Euclid)?.0)
            }
        }
    }

    /// `x · 2^bits` (one gate; `x` itself for 0 bits).
    fn scaled_by_power_of_two(&mut self, x: &Halo2BaseCell, bits: u32) -> Result<Halo2BaseCell, ProvingError> {
        if bits == 0 {
            return Ok(*x);
        }
        self.mul_by_constant(x, 1i128 << bits)
    }

    /// Float base, integer exponent, as the mock: `power = 1` multiplied by
    /// the base `|e|` times with the fixed-point product, then `1 / power`
    /// for a negative exponent. A constant exponent unrolls exactly that
    /// chain (the first product, `1 · base`, is the base itself). A runtime
    /// exponent is asserted into `[0, 8]` (larger ones have no witness), all
    /// nine powers are computed, and the result is the inner product of the
    /// indicators `[e == i]` with them: the bound leaves exactly one
    /// indicator equal to 1.
    pub(super) fn fixed_pow_int(&mut self, base: &Halo2BaseCell, exp: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        if self.is_const_cell(exp) {
            let e = kernel::fp_to_i64(exp.value);
            let power = self.power_chain(base, e.unsigned_abs())?;
            if e >= 0 {
                return Ok(power);
            }
            let one = self.constant_float(1.0)?;
            return self.fixed_div(&one, &power);
        }
        const MAX_EXP: u64 = 8;
        self.assert_less(exp, Operand::Constant(MAX_EXP as i128 + 1))?;
        let mut choices = Vec::with_capacity(MAX_EXP as usize + 1);
        let mut power = self.constant_float(1.0)?;
        for i in 0..=MAX_EXP {
            if i > 0 {
                power = if i == 1 { *base } else { self.fixed_mul(&power, base)? };
            }
            let index = self.constant_int(i as i64)?;
            choices.push((self.eq_i(exp, &index)?, power));
        }
        Ok(self.one_hot_combination(&choices))
    }

    /// `base^e` by `e - 1` fixed-point products, left to right (`1.0` for
    /// `e = 0`), the mock's rounding.
    fn power_chain(&mut self, base: &Halo2BaseCell, e: u64) -> Result<Halo2BaseCell, ProvingError> {
        if e == 0 {
            return self.constant_float(1.0);
        }
        let mut power = *base;
        for _ in 1..e {
            power = self.fixed_mul(&power, base)?;
        }
        Ok(power)
    }

    /// Float base and float exponent: supported for a constant exponent
    /// with an integer value (then the integer-exponent lowering); any other
    /// exponent is rejected with an error naming the operation.
    pub(super) fn fixed_pow(&mut self, base: &Halo2BaseCell, exp: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        match self.factor_of(exp) {
            Factor::Constant { m, s } if s == self.params.precision_bits => {
                let e = self.constant_int(m as i64)?;
                self.fixed_pow_int(base, &e)
            }
            _ => Err(ProvingError::synthesis(
                "halo2-base backend does not support `**` with a runtime or fractional float exponent",
            )),
        }
    }

    pub(super) fn sqrt_nonneg(&mut self, x: &Halo2BaseCell, x_bits: usize) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let scale = kernel::scale_fp(precision_bits);
        let big_x = self.gate_cell(x.value * scale, |gate, ctx| {
            gate.mul(ctx, x.quantum(), QuantumCell::Constant(scale))
        });
        self.isqrt(&big_x, x_bits + precision_bits as usize)
    }


    /// `IR::SqrtProductIF`: `floor(sqrt(b·c) * 2^P)` of two integers, admitted
    /// for `0 <= b < 2^64` and `0 <= n = b·c < 2^m` (field values), with
    /// `m = kernel::sqrt_int_bits(P)` (62 at `P = 32`).
    ///
    /// Soundness: the range checks of `b` to 64 bits and of `n` to exactly
    /// `m` bits (each skipped when the range is already proven inside) are
    /// the operator's domain: outside it there is no witness, as
    /// `kernel::fp_sqrt_product_int` says. Inside, `N = n * 2^(2P) < 2^126`
    /// and `isqrt` pins its floor root `y`, with `y^2 <= N < 2^126`, so
    /// `y < 2^63` (recorded: a later division by it knows its divisor is
    /// non-negative and in the window).
    pub(super) fn sqrt_product_int(&mut self, b: &Halo2BaseCell, c: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        if kernel::fp_sqrt_product_int(b.value, c.value, precision_bits).is_none() {
            let (bv, nv) = (kernel::fp_to_i128(b.value), kernel::fp_to_i128(b.value * c.value));
            self.note_unsat(move || format!("sqrt of an integer product: b = {bv:?}, b * c = {nv:?} outside the domain"));
        }
        self.ensure_bits(b, 64);
        let n = self.mul_i(b, c)?;
        let domain_bits = kernel::sqrt_int_bits(precision_bits) as usize;
        self.ensure_bits(&n, domain_bits);
        let scale = kernel::i128_to_fp(1i128 << (2 * precision_bits));
        let big_n = self.gate_cell(n.value * scale, |gate, ctx| {
            gate.mul(ctx, n.quantum(), QuantumCell::Constant(scale))
        });
        let root = self.isqrt(&big_n, domain_bits + 2 * precision_bits as usize)?;
        self.prove_range(&root, Interval { lo: 0, hi: (1i128 << 63) - 1 });
        Ok(root)
    }

    /// Constrain `x` into `[0, 2^bits)` (exactly) unless its range is already
    /// proven inside.
    fn ensure_bits(&mut self, x: &Halo2BaseCell, bits: usize) {
        let domain = Interval { lo: 0, hi: (1i128 << bits) - 1 };
        if !self.proven_range(x).is_some_and(|r| r.within(&domain)) {
            self.range_check(x, bits);
        }
    }

    /// `floor(sqrt(n))` for a cell `n` the caller has proven to lie in
    /// `[0, 2^n_bits)` (`n_bits <= 126`).
    ///
    /// Soundness: with the hint `y`, `d = n - y^2` (gate) and `s = 2y - d`,
    /// the checks `d ∈ [0, 2^wd)` and `s ∈ [0, 2^wd)` (`wd >= n_bits/2 + 2`)
    /// pin `y` without a range check of its own. In the field
    /// `2y = d + s ∈ [0, 2^(wd+1))`, so `y` is either the integer `(d + s)/2
    /// < 2^wd` or, when `d + s` is odd, `(d + s + p)/2`. In the second case
    /// `y^2 ≡ ((d + s)^2 + 3p)/4 (mod p)` (BN254's `p ≡ 1 mod 4`), a value in
    /// `[3p/4, p)`, so `d ≡ n - y^2` would lie near `p/4`, outside its
    /// check: impossible. In the first case `y^2 + d < 2^(2wd+1)`, far below
    /// `p`, so `n = y^2 + d` holds over the integers, and `0 <= d <= 2y`
    /// gives `y^2 <= n < (y + 1)^2`: `y` is the floor root of `n`, a function
    /// of `n`. Then `y < 2^ceil(n_bits/2)` (recorded as `y`'s range).
    pub(super) fn isqrt(&mut self, n: &Halo2BaseCell, n_bits: usize) -> Result<Halo2BaseCell, ProvingError> {
        let y_value = kernel::fp_to_i128(n.value)
            .filter(|v| *v >= 0)
            .map_or(0, |v| kernel::isqrt_u128(v as u128) as i128);
        let y = self.hint(kernel::i128_to_fp(y_value));
        let d = self.gate_cell(n.value - y.value * y.value, |gate, ctx| {
            gate.sub_mul(ctx, n.quantum(), y.quantum(), y.quantum())
        });
        let root_bits = n_bits.div_ceil(2);
        let wd = self.lookup_aligned(root_bits + 2);
        self.range_check(&d, wd);
        let two_y = self.add_i(&y, &y)?;
        let slack = self.sub_i(&two_y, &d)?;
        self.range_check(&slack, wd);
        self.prove_range(&y, Interval { lo: 0, hi: (1i128 << root_bits) - 1 });
        Ok(y)
    }


    /// Horner evaluation with leading-first Q coefficients, every product
    /// rounded down by `mul_f`: the algorithm of `kernel::horner_eval`.
    pub(super) fn poly_eval_horner(&mut self, x: &Halo2BaseCell, coefs: &[Fp]) -> Result<Halo2BaseCell, ProvingError> {
        let (first, rest) = coefs
            .split_first()
            .ok_or_else(|| ProvingError::synthesis("empty polynomial"))?;
        let mut acc = self.const_cell(*first);
        for coef in rest {
            let prod = self.mul_f(x, &acc)?;
            let coef = kernel::fp_to_i128(*coef).ok_or_else(|| ProvingError::synthesis("polynomial coefficient out of range"))?;
            acc = self.add_constant(&prod, coef);
        }
        Ok(acc)
    }


    /// `2^e` for a cell `e`, decomposed into `bits` booleans by `num_to_bits`
    /// (which recomposes them into `e`, so there is no witness for
    /// `e >= 2^bits`): the product of the constants `2^(2^j)` the bits
    /// select, `1 + b_j·(2^(2^j) - 1)` each. `bits <= 7`, so the constants
    /// and the product stay below `2^128`.
    pub(super) fn power_of_two(&mut self, e: &Halo2BaseCell, bits: usize) -> Halo2BaseCell {
        assert!((1..=7).contains(&bits), "power_of_two: 1 to 7 bits");
        let e_cell = self.assigned(e);
        let decomposed = self.gate.num_to_bits(self.builder.main(0), e_cell, bits);
        let mut power: Option<Halo2BaseCell> = None;
        for (j, assigned) in decomposed.into_iter().enumerate() {
            let value = *assigned.value();
            let bit = self.wrap(assigned, value);
            self.mark_bit(&bit);
            let step = kernel::i128_to_fp((1i128 << (1u32 << j)) - 1);
            let factor = self.gate_cell(Fp::one() + bit.value * step, |gate, ctx| {
                gate.mul_add(ctx, bit.quantum(), QuantumCell::Constant(step), QuantumCell::Constant(Fp::one()))
            });
            power = Some(match power {
                None => factor,
                Some(p) => self.gate_cell(p.value * factor.value, |gate, ctx| gate.mul(ctx, p.quantum(), factor.quantum())),
            });
        }
        let power = power.expect("at least one bit");
        let max_exponent = (1u32 << bits) - 1;
        let hi = if max_exponent >= 127 { i128::MAX } else { 1i128 << max_exponent };
        self.prove_range(&power, Interval { lo: 1, hi });
        power
    }

    /// `base^e` for a constant `e >= 0`, by square-and-multiply.
    pub(super) fn pow_const(&mut self, base: &Halo2BaseCell, e: u64) -> Result<Halo2BaseCell, ProvingError> {
        let mut acc = self.constant_int(1)?;
        let mut power = *base;
        let mut rest = e;
        while rest > 0 {
            if rest & 1 == 1 {
                acc = self.mul_i(&acc, &power)?;
            }
            rest >>= 1;
            if rest > 0 {
                power = self.mul_i(&power, &power)?;
            }
        }
        Ok(acc)
    }
}

/// The quotients `(a - r) / c` can take for `a` in `range` and `|r| < |c|`:
/// within one of `a / c`, so between `floor(min) - 1` and `floor(max) + 2`.
fn quotient_range(range: Interval, c: i128) -> Option<Interval> {
    if c == 0 {
        return None;
    }
    let floor = |a: i128| kernel::div_rem_or_zero(a, c, Rounding::Floor).0;
    let (x, y) = (floor(range.lo), floor(range.hi));
    Interval::new(x.min(y) - 1, x.max(y) + 2)
}
