//! `exp`, `log` and `tanh` on Q32 values.
//!
//! CONVENTIONS fixes the rounding of `sin` and `cos` only. For these functions
//! the rule (NPB_BASELINES.md, section 5, rule 5) is a value within the case's
//! tolerance (1e-6 absolute plus 1e-6 relative) of the real function for every
//! input whose result fits Q32, by the cheapest sound construction. Each one
//! reduces its argument to a short interval, where a minimax polynomial is
//! accurate, and evaluates that polynomial exactly in the field: the
//! coefficients and the reduced argument are integers, so the polynomial's
//! value is an exact integer at a known scale, and one floor division (a
//! range-checked remainder) brings it back to Q32. The error is the
//! polynomial's plus a single floor, measured by an integer emulation of
//! these exact steps on 3e5 arguments and checked by the tests:
//!
//! * `exp`: `x = k ln 2 + r` with `k` the nearest integer to `x / ln 2` and
//!   `|r| <= ln 2 / 2`, `exp(x) = 2^k exp(r)`, degree-5 polynomial. Relative
//!   error below 9.5e-8, plus the final floor (2^-32). An argument below
//!   `-32 ln 2` is clamped there: `exp` of it is below 2^-32 and the clamped
//!   result is at most 1 unit of 2^-32.
//! * `log`: `x = 2^j m` with `m` in `[1/sqrt 2, sqrt 2)`, then
//!   `log m = 2 atanh(s)` with `s = (m - 1) / (m + 1)` in `[-0.172, 0.172]`,
//!   an odd degree-5 polynomial in `s`. Absolute error below 2.3e-8.
//! * `tanh`: `tanh(|x|) = (1 - e) / (1 + e)` with `e = exp(-2 |x|)` from the
//!   `exp` construction, then the sign. Absolute error below 4.6e-8.
//!
//! An argument whose result is not finite or leaves the window has no
//! witness: `log` of zero or of a negative value, `exp` of an argument at or
//! above `31 ln 2` (its result reaches 2^31). Within a relative 8e-8 of that
//! bound the polynomial's own error can take the result over 2^31, so `exp`
//! has no proof for arguments in `[31 ln 2 - 1e-7, 31 ln 2)` either.
//!
//! The polynomials are weighted minimax fits (Remez exchange): exp's minimizes
//! the relative error on `[-ln 2 / 2, ln 2 / 2]`, log's the absolute error of
//! `log((1 + s) / (1 - s))` on `[0, 0.172]` (odd, so the fit is in `s^2`).

use halo2_base::utils::{biguint_to_fe, fe_to_biguint};

use super::*;

/// `round(ln 2 * 2^32)`.
const LN2_Q: i128 = 2_977_044_472;
/// `floor(LN2_Q / 2)`: the reduced argument of `exp` lies in
/// `[-HALF_LN2_Q, LN2_Q - HALF_LN2_Q)`.
const HALF_LN2_Q: i128 = LN2_Q / 2;
/// `exp` clamps its argument at `-32 ln 2`, where the result is 2^-32.
const EXP_MIN_Q: i128 = -32 * LN2_Q;
/// Bits of `k + 32`, the power of two of `exp` (`k` in `[-32, 31]`).
const EXP_POW_BITS: usize = 6;
/// The degree-5 minimax polynomial for `exp(r)` on `[-ln 2 / 2, ln 2 / 2]`
/// (relative error 7.5e-8; 9.5e-8 with the coefficients rounded),
/// coefficients `round(c * 2^24)`, constant first: 1.0000000717,
/// 0.9999996920, 0.4999889485, 0.1666757473, 0.0419153820, 0.0082976551.
const EXP_COEFS: [i128; 6] = [16_777_217, 16_777_211, 8_388_423, 2_796_355, 703_223, 139_212];
/// Scale of the exact polynomial: 24 coefficient bits plus 32 bits for each
/// of the five powers of `r`; 184 is 23 limbs of the 8-bit lookup.
const EXP_SHIFT: usize = 184;
/// `ceil(2^63 / sqrt 2)`: `log` normalizes its argument to `N` in `[L, 2L)`.
const LOG_L: u128 = 6_521_908_912_666_391_107;
/// The odd degree-5 minimax polynomial for `log((1 + s) / (1 - s))` on
/// `|s| <= 0.172`, as `s (c0 + c1 s^2 + c2 s^4)` (absolute error 2.1e-8),
/// coefficients `round(c * 2^32)`: 2.0000008370, 0.6664407804, 0.4151770601.
const LOG_COEFS: [i128; 3] = [8_589_938_187, 2_862_341_357, 1_783_171_895];
/// Scale of the exact polynomial `C0 s 2^128 + C1 s^3 2^64 + C2 s^5`: 32
/// coefficient bits plus 32 bits for each of five factors `s`; the result is
/// the polynomial at scale 2^32, so the remainder has 192 - 32 bits.
const LOG_SHIFT: usize = 160;
/// Width of the remainder range checks of `log`'s and `tanh`'s divisions:
/// enough for the divisor, a multiple of the 8-bit lookup.
const LOG_DIV_BITS: usize = 72;
const TANH_DIV_BITS: usize = 40;

/// `2^shift` as a field element.
fn pow2(shift: usize) -> Fr {
    Fr::from(2).pow_vartime([shift as u64])
}

/// `c * 2^shift` as a field element.
fn scaled(c: i128, shift: usize) -> Fr {
    fe(c) * pow2(shift)
}

/// `(floor(v / 2^shift), v mod 2^shift)` of a field element read as an integer
/// in `[0, p)`.
fn split_pow2(v: &Fr, shift: usize) -> (Fr, Fr) {
    let n = fe_to_biguint(v);
    let q = &n >> shift;
    let r = &n - (&q << shift);
    (biguint_to_fe(&q), biguint_to_fe(&r))
}

impl Q32 {
    // -------------------------------------------------------------------- exp

    /// `np.exp(x)`.
    ///
    /// Constraints: `x` is clamped at `-32 ln 2` (one 64-bit comparison), then
    /// [`Q32::exp_clamped`]; the result is window-checked, so an argument whose
    /// result leaves the window has no witness.
    pub fn exp(&self, ctx: &mut Ctx, x: I64) -> I64 {
        let low = self.lt_const(ctx, x, EXP_MIN_Q);
        let z = self.gate.select(ctx, Constant(fe(EXP_MIN_Q)), x.cell, low);
        let y = self.exp_clamped(ctx, z);
        self.bound(ctx, y)
    }

    /// `np.exp(-g)` for a `g` the circuit already proved to be an integer in
    /// `[0, 2^64)`, such as a gap `m - x` of [`Q32::max_of`] (softmax's
    /// `x - max(x)`). Cheaper than [`Q32::exp`]: the clamp compares `g` itself,
    /// and the result, at most 1, needs a 33-bit check, not the window.
    pub fn exp_neg(&self, ctx: &mut Ctx, g: Cell) -> I64 {
        // g < 2^64, as is_less_than requires.
        let far = self.range.is_less_than(ctx, Constant(fe(-EXP_MIN_Q)), g, 64);
        let minus_g = self.gate.neg(ctx, g);
        let z = self.gate.select(ctx, Constant(fe(EXP_MIN_Q)), minus_g, far);
        self.exp_at_most_one(ctx, z)
    }

    /// `exp(z)` for `z` in `[-32 ln 2, 0]`: [`Q32::exp_clamped`] with the
    /// result checked to `[0, 2^33)` (it is at most `2^32 (1 + 8e-8)`), which
    /// puts it in the window.
    fn exp_at_most_one(&self, ctx: &mut Ctx, z: Cell) -> I64 {
        let y = self.exp_clamped(ctx, z);
        self.range.range_check(ctx, y, 33);
        I64 { cell: y, offset: self.gate.add(ctx, y, Constant(Fr::from_u128(OFFSET))) }
    }

    /// `exp(z)` for a `z` the circuit already holds in `[-32 ln 2, 2^63)`.
    /// The result `y` is unbounded: the caller checks it to the window (or
    /// tighter), which with the remainder check of [`Q32::exp_scale_with`]
    /// pins it.
    ///
    /// Reduction: `t = z + 32 ln 2 + ln 2 / 2 = e LN2_Q + R` with `e` six bits
    /// and `R` in `[0, LN2_Q)` ([`Q32::exp_reduce_with`]), so `e = k + 32` for
    /// the nearest integer `k` to `z / ln 2` and `r = R - HALF_LN2_Q` is the
    /// reduced argument. Then `exp(z) = 2^(e - 32) exp(r)`:
    /// `y = floor(P(r) 2^e / 2^184)` with `P` the exact polynomial at scale
    /// 2^184 ([`Q32::exp_poly`]).
    fn exp_clamped(&self, ctx: &mut Ctx, z: Cell) -> Cell {
        let t = self.gate.add(ctx, z, Constant(fe(-EXP_MIN_Q + HALF_LN2_Q)));
        let tv = signed(t.value());
        let (bits, r) = self.exp_reduce_with(ctx, t, fe(tv.div_euclid(LN2_Q)), fe(tv.rem_euclid(LN2_Q)));
        let p = self.exp_poly(ctx, r);
        let power = self.pow2_of_bits(ctx, &bits);
        let x = self.gate.mul(ctx, p, power);
        let (y, rem) = split_pow2(x.value(), EXP_SHIFT);
        self.exp_scale_with(ctx, x, y, rem)
    }

    /// `t = e LN2_Q + R` for the hints `e` (six bits, returned) and `R`, with
    /// `R` in `[0, LN2_Q)` (two 32-bit checks); returns the bits of `e` and
    /// `r = R - HALF_LN2_Q`. `e LN2_Q + R` is an integer in `[0, 2^38)`, so for
    /// `t` in `[0, 64 LN2_Q)` the field identity is the integer floor
    /// decomposition and `(e, R)` is unique; a larger `t` (a result at or above
    /// 2^31.5) has no witness.
    fn exp_reduce_with(&self, ctx: &mut Ctx, t: Cell, e: Fr, rem: Fr) -> (Vec<Cell>, Cell) {
        let e = ctx.load_witness(e);
        let bits = self.gate.num_to_bits(ctx, e, EXP_POW_BITS);
        let rem = ctx.load_witness(rem);
        let recomposed = self.gate.mul_add(ctx, e, Constant(fe(LN2_Q)), rem);
        ctx.constrain_equal(&recomposed, &t);
        self.range.range_check(ctx, rem, 32);
        let gap = self.gate.sub(ctx, Constant(fe(LN2_Q - 1)), rem);
        self.range.range_check(ctx, gap, 32);
        (bits, self.gate.sub(ctx, rem, Constant(fe(HALF_LN2_Q))))
    }

    /// `sum_i EXP_COEFS[i] r^i 2^(32 (5 - i))`: exp's polynomial at the reduced
    /// argument `r` (a Q32 code, `|r| < 2^31`), exactly, at scale 2^184; Horner's
    /// rule with one gate per coefficient and no rounding. Its value lies in
    /// `(0, 1.4143 * 2^184)`.
    fn exp_poly(&self, ctx: &mut Ctx, r: Cell) -> Cell {
        let degree = EXP_COEFS.len() - 1;
        let mut acc = ctx.load_constant(fe(EXP_COEFS[degree]));
        for i in (0..degree).rev() {
            acc = self.gate.mul_add(ctx, acc, r, Constant(scaled(EXP_COEFS[i], 32 * (degree - i))));
        }
        acc
    }

    /// `2^e` from the little-endian bits of `e`: the product of
    /// `1 + b_i (2^(2^i) - 1)`.
    fn pow2_of_bits(&self, ctx: &mut Ctx, bits: &[Cell]) -> Cell {
        let mut power: Option<Cell> = None;
        for (i, b) in bits.iter().enumerate() {
            let factor = self.gate.mul_add(ctx, *b, Constant(pow2(1 << i) - Fr::ONE), Constant(Fr::ONE));
            power = Some(match power {
                None => factor,
                Some(p) => self.gate.mul(ctx, p, factor),
            });
        }
        power.expect("at least one bit")
    }

    /// `x = y 2^184 + rem` for the hints `y` (returned unbounded: the caller
    /// checks it to the window or tighter) and `rem` in `[0, 2^184)`.
    /// `x = P 2^e` is an integer in `[0, 1.4143 * 2^247)`, and with `y` in the
    /// window `y 2^184 + rem` is an integer in `[-2^247, 2^247 + 2^184)`; the
    /// two differ by less than `p`, so the field identity is the integer floor
    /// decomposition (in particular `y >= 0`) and `(y, rem)` is unique.
    fn exp_scale_with(&self, ctx: &mut Ctx, x: Cell, y: Fr, rem: Fr) -> Cell {
        ctx.assign_region([Witness(rem), Constant(pow2(EXP_SHIFT)), Witness(y), Existing(x)], [0]);
        let rem = ctx.get(-4);
        let y = ctx.get(-2);
        self.range.range_check(ctx, rem, EXP_SHIFT);
        y
    }

    // -------------------------------------------------------------------- log

    /// `np.log(x)` for `x > 0`; zero and negative arguments have no witness
    /// (NumPy returns `-inf` and `nan`).
    pub fn log(&self, ctx: &mut Ctx, x: I64) -> I64 {
        let xv = x.value();
        let j = (0..64).find(|j| xv > 0 && (xv << j) as u128 >= LOG_L).unwrap_or(0);
        let n = (xv.max(0) as u128) << j;
        let (num, den) = ((n as i128 - (1 << 63)) * SCALE, n as i128 + (1 << 63));
        let s = num.div_euclid(den);
        self.log_with(ctx, x, fe(j as i128), fe(s), fe(num.rem_euclid(den)))
    }

    /// Constraints, with the hints `j` (six bits), `s` and its remainder:
    ///
    /// * `N = x 2^j` in `[L, 2L)`, `L = ceil(2^63 / sqrt 2)` (two 64-bit checks).
    ///   `x` is in the window and `2^j <= 2^63`, so `x 2^j` is an integer of
    ///   magnitude below 2^126: a non-positive `x` gives `p - |x 2^j|`, which
    ///   fails the checks, and a positive one fixes `j`. Then
    ///   `x = m 2^(31 - j)` with `m = N / 2^63` in `[1/sqrt 2, sqrt 2)`.
    /// * `s = floor((N - 2^63) 2^32 / (N + 2^63))`, i.e. `(m - 1) / (m + 1)` in
    ///   Q32: remainder and divisor gap in `[0, 2^72)`, `s + 2^31` in
    ///   `[0, 2^32)`; `|s (N + 2^63) + rem| < 2^97`, so `(s, rem)` is unique.
    /// * `log m = floor(P / 2^160)` for the exact polynomial
    ///   `P = C0 s 2^128 + C1 s^3 2^64 + C2 s^5` ([`Q32::log_scale_with`]).
    /// * `log x = log m + (31 - j) LN2_Q`; the result lies in the window by
    ///   construction (`|log m| < 2^31`, `|31 - j| <= 32`).
    fn log_with(&self, ctx: &mut Ctx, x: I64, j: Fr, s: Fr, rem: Fr) -> I64 {
        let j = ctx.load_witness(j);
        let bits = self.gate.num_to_bits(ctx, j, EXP_POW_BITS);
        let power = self.pow2_of_bits(ctx, &bits);
        let n = self.gate.mul(ctx, x.cell, power);
        let above = self.gate.sub(ctx, n, Constant(Fr::from_u128(LOG_L)));
        self.range.range_check(ctx, above, 64);
        let below = self.gate.sub(ctx, Constant(Fr::from_u128(2 * LOG_L - 1)), n);
        self.range.range_check(ctx, below, 64);

        let den = self.gate.add(ctx, n, Constant(Fr::from_u128(OFFSET)));
        let num = self.gate.mul_add(ctx, n, Constant(fe(SCALE)), Constant(-pow2(95)));
        let s = ctx.load_witness(s);
        let rem = ctx.load_witness(rem);
        let recomposed = self.gate.mul_add(ctx, s, den, rem);
        ctx.constrain_equal(&recomposed, &num);
        self.range.range_check(ctx, rem, LOG_DIV_BITS);
        let gap = self.gate.sub(ctx, den, rem);
        let gap = self.gate.sub(ctx, gap, Constant(Fr::ONE));
        self.range.range_check(ctx, gap, LOG_DIV_BITS);
        let s_offset = self.gate.add(ctx, s, Constant(fe(1 << 31)));
        self.range.range_check(ctx, s_offset, 32);

        let p = self.log_poly(ctx, s);
        let (log_m_offset, rem) = split_pow2(&(*p.value() + pow2(LOG_SHIFT + 31)), LOG_SHIFT);
        let log_m_offset = self.log_scale_with(ctx, p, log_m_offset, rem);
        // log m + (31 - j) LN2_Q = log_m_offset - 2^31 + 31 LN2_Q - j LN2_Q
        let partial = self.gate.mul_add(ctx, j, Constant(fe(-LN2_Q)), log_m_offset);
        let cell = self.gate.add(ctx, partial, Constant(fe(31 * LN2_Q - (1 << 31))));
        I64 { cell, offset: self.gate.add(ctx, cell, Constant(Fr::from_u128(OFFSET))) }
    }

    /// `C0 s 2^128 + C1 s^3 2^64 + C2 s^5` exactly (scale 2^192), by Horner's
    /// rule in `s^2`; its magnitude is below `0.35 * 2^192`.
    fn log_poly(&self, ctx: &mut Ctx, s: Cell) -> Cell {
        let s2 = self.gate.mul(ctx, s, s);
        let mut acc = ctx.load_constant(fe(LOG_COEFS[2]));
        acc = self.gate.mul_add(ctx, acc, s2, Constant(scaled(LOG_COEFS[1], 64)));
        acc = self.gate.mul_add(ctx, acc, s2, Constant(scaled(LOG_COEFS[0], 128)));
        self.gate.mul(ctx, acc, s)
    }

    /// `P + 2^191 = v 2^160 + rem` for the hints `v` in `[0, 2^32)` (returned;
    /// `v - 2^31 = floor(P / 2^160)`, `log m` in Q32) and `rem` in `[0, 2^160)`.
    /// `P + 2^191` lies in `(0, 2^192)` and `v 2^160 + rem` below `2^193`, so
    /// the identity is the integer floor decomposition and `(v, rem)` is unique.
    fn log_scale_with(&self, ctx: &mut Ctx, p: Cell, v: Fr, rem: Fr) -> Cell {
        let shifted = self.gate.add(ctx, p, Constant(pow2(LOG_SHIFT + 31)));
        ctx.assign_region([Witness(rem), Constant(pow2(LOG_SHIFT)), Witness(v), Existing(shifted)], [0]);
        let rem = ctx.get(-4);
        let v = ctx.get(-2);
        self.range.range_check(ctx, rem, LOG_SHIFT);
        self.range.range_check(ctx, v, 32);
        v
    }

    // ------------------------------------------------------------------- tanh

    /// `np.tanh(x)`.
    ///
    /// Constraints: the sign of `x` (one 64-bit comparison) and `|x|`; `2 |x|`
    /// clamped at `32 ln 2` (one comparison: `|x| <= 2^63` fits its 64-bit
    /// precondition); `e = exp(-2 |x|)` by [`Q32::exp_at_most_one`]; then
    /// [`Q32::tanh_ratio_with`] and the sign.
    pub fn tanh(&self, ctx: &mut Ctx, x: I64) -> I64 {
        let neg = self.is_neg(ctx, x);
        let minus_x = self.gate.neg(ctx, x.cell);
        let magnitude = self.gate.select(ctx, minus_x, x.cell, neg);
        let far = self.range.is_less_than(ctx, Constant(fe(-EXP_MIN_Q / 2)), magnitude, 64);
        let minus_twice = self.gate.mul(ctx, magnitude, Constant(-Fr::from(2)));
        let z = self.gate.select(ctx, Constant(fe(EXP_MIN_Q)), minus_twice, far);
        let e = self.exp_at_most_one(ctx, z);
        let (num, den) = ((SCALE - e.value()) * SCALE, SCALE + e.value());
        let t = self.tanh_ratio_with(ctx, e.cell, fe(num.div_euclid(den)), fe(num.rem_euclid(den)));
        let minus_t = self.gate.neg(ctx, t);
        let cell = self.gate.select(ctx, minus_t, t, neg);
        I64 { cell, offset: self.gate.add(ctx, cell, Constant(Fr::from_u128(OFFSET))) }
    }

    /// `t = floor((2^32 - e) 2^32 / (2^32 + e))` for `e` in `[0, 2^33)`, from
    /// the hints `t` and `rem`: remainder and divisor gap in `[0, 2^40)` (the
    /// divisor is below 2^34), `t + 2^32` in `[0, 2^40)` (`t` is slightly
    /// negative when `e` exceeds 2^32 by the polynomial's error). The terms
    /// stay below 2^75, so `(t, rem)` is unique. `t` lies in the window.
    fn tanh_ratio_with(&self, ctx: &mut Ctx, e: Cell, t: Fr, rem: Fr) -> Cell {
        let num = self.gate.mul_add(ctx, e, Constant(-fe(SCALE)), Constant(fe(SCALE * SCALE)));
        let den = self.gate.add(ctx, e, Constant(fe(SCALE)));
        let t = ctx.load_witness(t);
        let rem = ctx.load_witness(rem);
        let recomposed = self.gate.mul_add(ctx, t, den, rem);
        ctx.constrain_equal(&recomposed, &num);
        self.range.range_check(ctx, rem, TANH_DIV_BITS);
        let gap = self.gate.sub(ctx, den, rem);
        let gap = self.gate.sub(ctx, gap, Constant(Fr::ONE));
        self.range.range_check(ctx, gap, TANH_DIV_BITS);
        let t_offset = self.gate.add(ctx, t, Constant(fe(SCALE)));
        self.range.range_check(ctx, t_offset, TANH_DIV_BITS);
        t
    }
}

#[cfg(test)]
mod tests {
    //! Accuracy against `f64` on arguments across each domain, and malicious
    //! hints: every alternative a prover could load for a witnessed value (a
    //! neighbouring quotient, a non-integer field solution) is rejected.
    use super::*;
    use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
    use halo2_base::halo2_proofs::dev::MockProver;

    const K: u32 = 12;

    /// Whether the circuit built by `f` is satisfied (as in `q32::tests`).
    fn accepts(f: impl FnOnce(&Q32, &mut Ctx)) -> bool {
        let mut builder = BaseCircuitBuilder::<Fr>::from_stage(CircuitBuilderStage::Mock);
        builder.set_k(K as usize);
        builder.set_lookup_bits(8);
        builder.set_instance_columns(1);
        let q = Q32::new(&builder);
        f(&q, builder.main(0));
        builder.calculate_params(Some(20));
        let ok = MockProver::run(K, &builder, vec![vec![]]).unwrap().verify().is_ok();
        builder.clear();
        ok
    }

    /// The field solution `(n - r) / d` a prover gets by picking remainder `r`.
    fn field_quotient(n: i128, r: i128, d: i128) -> Fr {
        (fe(n) - fe(r)) * fe(d).invert().unwrap()
    }

    /// Q32 arguments of exp: the clamp region, both sides of zero, the
    /// reduction boundaries, up to just below the top of the domain.
    fn exp_arguments() -> Vec<i128> {
        let mut xs: Vec<i128> = [-2147483648.0, -1000.0, -30.0, -22.5, -22.0, -5.3, -1.0, -0.3466, 0.0, 1e-9, 0.25,
            0.3466, 0.3467, 1.0, 2.0, 7.7, 15.0, 21.0]
            .iter()
            .map(|v| quantize(*v))
            .collect();
        xs.extend([EXP_MIN_Q, EXP_MIN_Q - 1, EXP_MIN_Q + 1, HALF_LN2_Q, HALF_LN2_Q - 1, 31 * LN2_Q - 1_000]);
        xs
    }

    #[test]
    fn exp_matches_f64() {
        for x in exp_arguments() {
            assert!(accepts(|q, ctx| {
                let xb = q.load_int_bounded(ctx, x as i64);
                let y = dequantize(q.exp(ctx, xb).cell.value());
                let want = (x as f64 / SCALE as f64).exp();
                assert!((y - want).abs() <= 9.5e-8 * want + 2.4e-10, "exp({x}) = {y}, want {want}");
                if x <= 0 {
                    let g = ctx.load_witness(fe(-x));
                    let yn = dequantize(q.exp_neg(ctx, g).cell.value());
                    assert_eq!(yn, y, "exp_neg({})", -x);
                }
            }), "exp({x})");
        }
    }

    #[test]
    fn exp_overflow_has_no_witness() {
        for x in [31 * LN2_Q, quantize(21.5), quantize(1e6), i64::MAX as i128] {
            assert!(!accepts(|q, ctx| {
                let xb = q.load_int_bounded(ctx, x as i64);
                q.exp(ctx, xb);
            }), "exp({x})");
        }
    }

    #[test]
    fn exp_reduction_is_pinned() {
        let x = quantize(1.7);
        let t = x - EXP_MIN_Q + HALF_LN2_Q;
        let (e, rem) = (t.div_euclid(LN2_Q), t.rem_euclid(LN2_Q));
        let run = |eh: Fr, rh: Fr| {
            accepts(move |q, ctx| {
                let t = ctx.load_witness(fe(t));
                q.exp_reduce_with(ctx, t, eh, rh);
            })
        };
        assert!(run(fe(e), fe(rem)));
        assert!(!run(fe(e + 1), fe(rem - LN2_Q)));
        assert!(!run(fe(e - 1), fe(rem + LN2_Q)));
        assert!(!run(field_quotient(t, rem + 1, LN2_Q), fe(rem + 1)));
        // A remainder in [LN2_Q, 2^32) passes the 32-bit check, not the gap check.
        let t2 = 5 * LN2_Q + 1000;
        assert!(!accepts(move |q, ctx| {
            let t = ctx.load_witness(fe(t2));
            q.exp_reduce_with(ctx, t, fe(4), fe(1000 + LN2_Q));
        }));
    }

    #[test]
    fn exp_scale_is_pinned() {
        let x = fe(3) * pow2(EXP_SHIFT) + fe(12345);
        let run = |y: Fr, rem: Fr| {
            accepts(move |q, ctx| {
                let x = ctx.load_witness(x);
                let y = q.exp_scale_with(ctx, x, y, rem);
                q.bound(ctx, y);
            })
        };
        assert!(run(fe(3), fe(12345)));
        assert!(!run(fe(4), fe(12345) - pow2(EXP_SHIFT)));
        assert!(!run(fe(2), fe(12345) + pow2(EXP_SHIFT)));
        assert!(!run((x - fe(12346)) * pow2(EXP_SHIFT).invert().unwrap(), fe(12346)));
        // The decomposition of x - p (a negative y): its y is below -2^63.
        let (y_neg, rem_neg) = split_pow2(&-x, EXP_SHIFT);
        assert!(!run(-y_neg - Fr::ONE, pow2(EXP_SHIFT) - rem_neg));
    }

    /// Q32 arguments of log: the smallest code, both sides of 1 and of the
    /// normalization boundaries, up to the top of the window.
    fn log_arguments() -> Vec<i128> {
        let mut xs: Vec<i128> =
            [1e-9, 0.001, 0.5, 0.7071067, 0.7071068, 0.99999, 1.0, 1.00001, 1.4142135, 1.4142136, 2.0, 3.3, 1e6]
                .iter()
                .map(|v| quantize(*v))
                .collect();
        xs.extend([1, 2, 3, i64::MAX as i128]);
        xs
    }

    #[test]
    fn log_matches_f64() {
        for x in log_arguments() {
            assert!(accepts(|q, ctx| {
                let xb = q.load_int_bounded(ctx, x as i64);
                let y = dequantize(q.log(ctx, xb).cell.value());
                let want = (x as f64 / SCALE as f64).ln();
                assert!((y - want).abs() <= 2.3e-8, "log({x}) = {y}, want {want}");
            }), "log({x})");
        }
    }

    #[test]
    fn log_of_non_positive_has_no_witness() {
        for x in [0i128, -1, quantize(-2.5), i64::MIN as i128] {
            for j in [0i128, 1, 33, 63] {
                assert!(!accepts(|q, ctx| {
                    let xb = q.load_int_bounded(ctx, x as i64);
                    q.log_with(ctx, xb, fe(j), fe(0), fe(0));
                }), "log({x}) with j = {j}");
            }
        }
    }

    #[test]
    fn log_hints_are_pinned() {
        let x = quantize(3.3);
        let run = |dj: i128, ds: i128, drem: i128| {
            accepts(move |q, ctx| {
                let xb = q.load_int_bounded(ctx, x as i64);
                let j = (0..64).find(|j| (x << j) as u128 >= LOG_L).unwrap() as i128;
                let n = x << j;
                let (num, den) = ((n - (1 << 63)) * SCALE, n + (1 << 63));
                let (s, rem) = (num.div_euclid(den), num.rem_euclid(den));
                q.log_with(ctx, xb, fe(j + dj), fe(s + ds), fe(rem - ds * den + drem));
            })
        };
        assert!(run(0, 0, 0));
        assert!(!run(1, 0, 0) && !run(-1, 0, 0)); // the neighbouring normalizations
        assert!(!run(0, 1, 0) && !run(0, -1, 0)); // neighbouring quotients
        let p = fe(-12345) * pow2(LOG_SHIFT) + fe(678);
        let scale = |v: Fr, rem: Fr| {
            accepts(move |q, ctx| {
                let p = ctx.load_witness(p);
                q.log_scale_with(ctx, p, v, rem);
            })
        };
        let v = fe((1 << 31) - 12345);
        assert!(scale(v, fe(678)));
        assert!(!scale(v + Fr::ONE, fe(678) - pow2(LOG_SHIFT)));
        assert!(!scale(v - Fr::ONE, fe(678) + pow2(LOG_SHIFT)));
        let shifted = p + pow2(LOG_SHIFT + 31);
        assert!(!scale((shifted - fe(679)) * pow2(LOG_SHIFT).invert().unwrap(), fe(679)));
    }

    #[test]
    fn tanh_matches_f64() {
        let args = [-2147483648.0, -40.0, -11.1, -11.0, -3.0, -0.5, -1e-6, 0.0, 1e-9, 0.1, 0.7, 1.0, 2.5, 7.0, 11.2, 100.0];
        for v in args {
            assert!(accepts(|q, ctx| {
                let xb = q.load_float_bounded(ctx, v);
                let y = dequantize(q.tanh(ctx, xb).cell.value());
                assert!((y - v.tanh()).abs() <= 4.6e-8, "tanh({v}) = {y}");
            }), "tanh({v})");
        }
    }

    #[test]
    fn tanh_ratio_is_pinned() {
        let e = (SCALE * 3) / 7;
        let (num, den) = ((SCALE - e) * SCALE, SCALE + e);
        let (t, rem) = (num.div_euclid(den), num.rem_euclid(den));
        let run = |th: Fr, rh: Fr| {
            accepts(move |q, ctx| {
                let e = ctx.load_witness(fe(e));
                q.tanh_ratio_with(ctx, e, th, rh);
            })
        };
        assert!(run(fe(t), fe(rem)));
        assert!(!run(fe(t + 1), fe(rem - den)));
        assert!(!run(fe(t - 1), fe(rem + den)));
        assert!(!run(field_quotient(num, rem + 1, den), fe(rem + 1)));
    }

    /// Prints the cells of one call of each gadget (run with
    /// `cargo test --lib transcendental_cost -- --nocapture`).
    #[test]
    fn transcendental_cost() {
        type Gadget = fn(&Q32, &mut Ctx, I64);
        let gadgets: [(&str, Gadget); 4] = [
            ("exp", |q, ctx, x| {
                q.exp(ctx, x);
            }),
            ("exp_neg", |q, ctx, x| {
                q.exp_neg(ctx, x.cell);
            }),
            ("log", |q, ctx, x| {
                q.log(ctx, x);
            }),
            ("tanh", |q, ctx, x| {
                q.tanh(ctx, x);
            }),
        ];
        for (name, gadget) in gadgets {
            let mut b = BaseCircuitBuilder::<Fr>::from_stage(CircuitBuilderStage::Keygen);
            b.set_k(K as usize);
            b.set_lookup_bits(8);
            b.set_instance_columns(1);
            let q = Q32::new(&b);
            let ctx = b.main(0);
            let x = ctx.load_witness(fe(SCALE / 3));
            let x = I64 { cell: x, offset: q.gate.add(ctx, x, Constant(Fr::from_u128(OFFSET))) };
            let before = (ctx.advice.len(), 0usize);
            gadget(&q, ctx, x);
            let st = b.statistics();
            println!(
                "{name}: advice {} lookup {}",
                st.gate.total_advice_per_phase[0] - before.0,
                st.total_lookup_advice_per_phase[0]
            );
            b.clear();
        }
    }
}
