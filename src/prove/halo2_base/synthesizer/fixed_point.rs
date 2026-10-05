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
        let neg_x = self.gate_cell(-x.value, |gate, ctx| gate.neg(ctx, x.assigned));
        let magnitude = self.select_bit(&sign, &neg_x, x)?;
        Ok((sign, magnitude))
    }


    /// `(s, m)` with `s = [x < 0]` and `m = |x|`, for `x` in `[-2^64, 2^64)`.
    ///
    /// Soundness: `m = s ? -x : x` with a hinted sign, and the range check
    /// `m - s ∈ [0, 2^64)`. With the wrong sign `m` is the field negation of
    /// a small number (about p) and fails the check; for `x = 0` the sign
    /// `s = 1` gives `m - s = -1`, which fails too. So `s` and `m` are
    /// functions of `x`, and an `x` outside the window has no witness.
    pub(super) fn sign_and_abs(&mut self, x: &Halo2BaseCell) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
        if let Some(pair) = self.sign_abs_cache.get(&x.id) {
            return Ok(*pair);
        }
        let pair = if self.is_const_cell(x) {
            let negative = kernel::fp_is_negative(x.value);
            let sign = self.constant_bool(negative)?;
            let magnitude = self.const_cell(if negative { -x.value } else { x.value });
            (sign, magnitude)
        } else {
            let (sign, magnitude) = self.hinted_sign_and_magnitude(x)?;
            let shifted = self.sub_i(&magnitude, &sign)?;
            self.range_check(&shifted, 64);
            // m = (m - s) + s with m - s in [0, 2^64) and s a bit.
            self.prove_range(&magnitude, Interval { lo: 0, hi: 1 << 64 });
            (sign, magnitude)
        };
        self.sign_abs_cache.insert(x.id, pair);
        Ok(pair)
    }


    /// Constrain `0 <= x < n` for a cell `n`. Both `x` and `n - 1 - x` are
    /// range-checked to 64 bits, so `n - 1`, their sum, lies in `[0, 2^65)`
    /// over the integers: the check also proves `1 <= n <= 2^65`.
    pub(super) fn assert_below_cell(&mut self, x: &Halo2BaseCell, n: &Halo2BaseCell) {
        self.range_check(x, 64);
        let gap = self.gate_cell(n.value - Fp::one() - x.value, |gate, ctx| {
            let n_minus_x = gate.sub(ctx, n.assigned, x.assigned);
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
            None => {
                let b_is_zero = self.is_zero_bit(b);
                let b_nonzero = self.add_i(b, &b_is_zero)?;
                let a_masked = self.gate_cell(
                    (Fp::one() - b_is_zero.value) * a.value,
                    |gate, ctx| gate.mul_not(ctx, b_is_zero.assigned, a.assigned),
                );
                self.div_rem_nonzero(&a_masked, Divisor::NonZero(b_nonzero), rounding)
            }
        }
    }


    /// Constrained division by a divisor that is non-zero by construction.
    ///
    /// Constraints (A, B, Q, R are the integers the cells encode):
    ///   * `r = a - b*q` (one gate, exact in the field), `q` a hint;
    ///   * `q + 2^63 ∈ [0, 2^64)`: Q is in the signed 64-bit window;
    ///   * `0 <= r_adj < |B|`, where `r_adj = ±r` strips the convention's
    ///     remainder sign (Euclid: none; Floor: the divisor's; Trunc: the
    ///     dividend's, from `sign_and_abs`). For a constant divisor this is
    ///     [`Self::assert_less`]; for a cell it is
    ///     [`Self::assert_below_cell`] against `|b|`, which also proves
    ///     `1 <= |B| <= 2^65` and with it the hinted sign of `b`.
    ///
    /// Soundness: `|B·Q + R| < 2^65·2^63 + 2^65`, far below p/2, so
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
                (b.value, kernel::fp_to_i128(b.value), QuantumCell::Existing(b.assigned))
            }
        };
        let q_value = match (kernel::fp_to_i128(a.value), b_value) {
            (Some(av), Some(bv)) => kernel::div_rem_or_zero(av, bv, rounding).0,
            _ => 0,
        };
        let q = self.hint(kernel::i128_to_fp(q_value));
        let r = self.gate_cell(a.value - b_fp * q.value, |gate, ctx| {
            gate.sub_mul(ctx, a.assigned, b_quantum, q.assigned)
        });
        self.ensure_signed64(&q)?;

        // A dynamic divisor's hinted sign and magnitude, pinned by the
        // remainder bound below.
        let dynamic_sign = match divisor {
            Divisor::NonZero(b) => Some(self.hinted_sign_and_magnitude(&b)?),
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
        Ok((q, r))
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
        let q_value = kernel::fp_to_i128(raw.value).map_or(0, |v| v >> shift);
        let scale = kernel::i128_to_fp(1i128 << shift);
        let q = self.hint(kernel::i128_to_fp(q_value));
        let rem = self.gate_cell(raw.value - q.value * scale, |gate, ctx| {
            gate.sub_mul(ctx, raw.assigned, q.assigned, QuantumCell::Constant(scale))
        });
        self.range_check(&rem, shift as usize);
        self.ensure_signed64(&q)?;
        Ok(q)
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
        let out = self.gate_cell(x.value * factor, |gate, ctx| gate.mul(ctx, x.assigned, QuantumCell::Constant(factor)));
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
            match integer {
                Some((n, x)) => {
                    let term = self.mul_i(&n, &x)?;
                    exact = Some(match exact {
                        Some(sum) => self.add_i(&sum, &term)?,
                        None => term,
                    });
                }
                None => {
                    let acc = match raw {
                        Some(acc) => acc,
                        None => self.constant_int(0)?,
                    };
                    raw = Some(self.gate_cell(acc.value + a.value * b.value, |gate, ctx| {
                        gate.mul_add(ctx, a.assigned, b.assigned, acc.assigned)
                    }));
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

    pub(super) fn sqrt_nonneg(&mut self, x: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let scale = kernel::scale_fp(precision_bits);
        let big_x = self.gate_cell(x.value * scale, |gate, ctx| {
            gate.mul(ctx, x.assigned, QuantumCell::Constant(scale))
        });
        self.isqrt(&big_x, 64 + precision_bits as usize)
    }


    /// `floor(sqrt(n))` for a cell `n` the caller has proven to lie in
    /// `[0, 2^n_bits)`.
    ///
    /// Soundness: with the hint `y` and `d = n - y^2` (gate), the checks
    /// `y ∈ [0, 2^wy)`, `d ∈ [0, 2^wd)` and `2y - d ∈ [0, 2^wd)` give
    /// `n = y^2 + d` over the integers (both terms are far below p) and
    /// `0 <= d <= 2y`, i.e. `y^2 <= n < (y + 1)^2`. Exactly one `y >= 0`
    /// satisfies that, so `y` is a function of `n`. `wy` and `wd` are sized
    /// for the largest admissible `n`.
    pub(super) fn isqrt(&mut self, n: &Halo2BaseCell, n_bits: usize) -> Result<Halo2BaseCell, ProvingError> {
        let y_value = kernel::fp_to_i128(n.value)
            .filter(|v| *v >= 0)
            .map_or(0, |v| kernel::isqrt_u128(v as u128) as i128);
        let y = self.hint(kernel::i128_to_fp(y_value));
        let d = self.gate_cell(n.value - y.value * y.value, |gate, ctx| {
            gate.sub_mul(ctx, n.assigned, y.assigned, y.assigned)
        });
        let root_bits = n_bits.div_ceil(2);
        let wy = self.lookup_aligned(root_bits + 1);
        let wd = self.lookup_aligned(root_bits + 2);
        self.range_check(&y, wy);
        self.range_check(&d, wd);
        let two_y = self.add_i(&y, &y)?;
        let slack = self.sub_i(&two_y, &d)?;
        self.range_check(&slack, wd);
        Ok(y)
    }


    /// `c - x` for a constant `c` (one gate).
    pub(super) fn constant_minus(&mut self, c: i128, x: &Halo2BaseCell) -> Halo2BaseCell {
        let c_fp = kernel::i128_to_fp(c);
        self.gate_cell(c_fp - x.value, |gate, ctx| gate.sub(ctx, QuantumCell::Constant(c_fp), x.assigned))
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
            acc = self.gate_cell(prod.value + coef, |gate, ctx| {
                gate.add(ctx, prod.assigned, QuantumCell::Constant(*coef))
            });
        }
        Ok(acc)
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
