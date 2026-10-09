//! Frozen compiler arithmetic schedules for strict baseline comparisons.
//! Constants mirror prove/kernel.rs and benchmarking/noir_fpc/src/lib.nr.
//! Unlike the historical minimax shortcuts in transcendental.rs, each Horner
//! multiplication is rounded to Q32, and argument reductions are unchanged.
use super::*;

const EXP2_Q: [i128; 13] = [0, 2, 31, 437, 5676, 65510, 661577, 5726720,
    41309550, 238388332, 1031764991, 2977044472, 4294967296];
const LOG_SERIES_Q: [i128; 12] = [186737709, 204522252, 226050910, 252645135,
    286331153, 330382100, 390451572, 477218588, 613566757, 858993459, 1431655765, 4294967296];
const SIN_Q: [i128; 5] = [11136, -850500, 35789648, -715827153, 4294967215];
const ATAN_Q: [i128; 10] = [-7489138, 45873335, -131930687, 246804093,
    -359577404, 469877966, -612529925, 858917466, -1431653573, 4294967285];

impl Q32 {
    /// Frozen dynamic DivF: zero divisors map to zero. Source-language
    /// nonzero assertions, when present, must be imposed by the caller.
    /// The two 64-bit remainder checks constrain the divisor magnitude;
    /// unlike `div`, this does not first restrict the divisor to signed64.
    pub fn canonical_div_or_zero(
        &self, ctx: &mut Ctx, a: impl Into<QuantumCell<Fr>>, b: Cell,
    ) -> I64 {
        let zero = self.gate.is_zero(ctx, b);
        let divisor = self.add(ctx, b, zero);
        let numerator = self.gate.mul_not(ctx, zero, a);
        let negative = ctx.load_witness(fe((signed(divisor.value()) < 0) as i128));
        self.gate.assert_bit(ctx, negative);
        let opposite = self.gate.neg(ctx, divisor);
        let magnitude = self.gate.select(ctx, opposite, divisor, negative);
        // Decode after field multiplication. Accepted inputs can require a
        // 128-bit unsigned magnitude, so signed i128 multiplication is unsafe.
        let scaled = *numerator.value() * fe(SCALE);
        let small = |value: Fr| -> Option<u128> {
            let bytes = value.to_bytes_le();
            if bytes[16..].iter().any(|&byte| byte != 0) { return None; }
            Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
        };
        let (num, neg_num) = if let Some(value) = small(scaled) {
            (value, false)
        } else { (small(-scaled).unwrap_or(0), true) };
        let den = small(*magnitude.value()).unwrap_or(1).max(1);
        let quotient = num / den;
        let remainder = num % den;
        let round = neg_num && remainder != 0;
        let quotient = Fr::from_u128(quotient) + if round { Fr::ONE } else { Fr::ZERO };
        let quotient = if neg_num != (signed(divisor.value()) < 0) { -quotient } else { quotient };
        let remainder = if round { den - remainder } else { remainder };
        self.div_with(ctx, Existing(numerator), divisor, magnitude,
            quotient, Fr::from_u128(remainder))
    }

    fn canonical_horner(&self, ctx: &mut Ctx, x: Cell, coefficients: &[i128]) -> Cell {
        let mut acc = ctx.load_constant(fe(coefficients[0]));
        for &coefficient in &coefficients[1..] {
            let product = self.mul(ctx, acc, x);
            acc = self.add(ctx, product.cell, Constant(fe(coefficient)));
        }
        acc
    }

    fn canonical_pow2(&self, ctx: &mut Ctx, exponent: Cell, width: usize) -> Cell {
        let bits = self.gate.num_to_bits(ctx, exponent, width);
        let mut power = ctx.load_constant(Fr::ONE);
        for (i, bit) in bits.into_iter().enumerate() {
            let factor = self.gate.mul_add(ctx, bit,
                Constant(Fr::from(2).pow_vartime([1u64 << i]) - Fr::ONE), Constant(Fr::ONE));
            power = self.gate.mul(ctx, power, factor);
        }
        power
    }

    /// Exact kernel::fp_exp Q32 algorithm, including its intermediate window.
    pub fn canonical_exp(&self, ctx: &mut Ctx, x: I64) -> I64 {
        let y = self.mul(ctx, x.cell, Constant(fe(6196328019)));
        let (integer, fraction) = self.floor_div_const(ctx, y.cell, SCALE as u64);
        let mantissa = self.canonical_horner(ctx, fraction, &EXP2_Q);
        let exponent = self.add(ctx, integer.cell, Constant(fe(48)));
        let exponent = self.bound(ctx, exponent);
        let underflow = self.is_neg(ctx, exponent);
        let kept = self.gate.mul_not(ctx, underflow, exponent.cell);
        let power = self.canonical_pow2(ctx, kept, 7);
        let raw = self.gate.mul(ctx, mantissa, power);
        let (scaled, _) = self.floor_div_const(ctx, raw, 1u64 << 48);
        let cell = self.gate.mul_not(ctx, underflow, scaled.cell);
        self.bound(ctx, cell)
    }

    /// Same arithmetic as exp(-gap); no alternate underflow clamp/polynomial.
    pub fn canonical_exp_neg(&self, ctx: &mut Ctx, gap: Cell) -> I64 {
        let negative = self.gate.neg(ctx, gap);
        let negative = self.bound(ctx, negative);
        self.canonical_exp(ctx, negative)
    }

    /// Exact kernel::fp_log: normalize to [1,2), 12-term atanh series and
    /// independently rounded k*ln(2) from Q64. Every hint is constrained.
    pub fn canonical_log(&self, ctx: &mut Ctx, x: impl Into<QuantumCell<Fr>>) -> I64 {
        let x = x.into();
        let positive = self.sub(ctx, x, Constant(Fr::ONE));
        self.range.range_check(ctx, positive, 64);
        let value = signed(x.value()).max(1) as u128;
        let j = (127 - value.leading_zeros()) as usize;
        let mantissa = if j >= 32 { value >> (j - 32) } else { value << (32 - j) };
        let exponent = ctx.load_witness(fe(j as i128));
        let mantissa = ctx.load_witness(fe(mantissa as i128));
        let offset = self.sub(ctx, mantissa, Constant(fe(SCALE)));
        self.range.range_check(ctx, offset, 32);
        let power = self.canonical_pow2(ctx, exponent, 6);
        let shifted = self.gate.mul(ctx, x, Constant(fe(SCALE)));
        let remainder = self.gate.sub_mul(ctx, shifted, mantissa, power);
        self.range.range_check(ctx, remainder, 64);
        let gap = self.sub(ctx, power, remainder);
        let gap = self.sub(ctx, gap, Constant(Fr::ONE));
        self.range.range_check(ctx, gap, 64);
        let numerator = self.sub(ctx, mantissa, Constant(fe(SCALE)));
        let denominator = self.add(ctx, mantissa, Constant(fe(SCALE)));
        let s = self.div_by_abs(ctx, numerator, denominator, denominator);
        let squared = self.mul(ctx, s.cell, s.cell);
        let series = self.canonical_horner(ctx, squared.cell, &LOG_SERIES_Q);
        let half = self.mul(ctx, s.cell, series);
        let raw = self.gate.mul_add(ctx, exponent, Constant(fe(12786308645202655660)),
            Constant(fe((1i128 << 31) - 32 * 12786308645202655660i128)));
        let (k_ln2, _) = self.floor_div_const(ctx, raw, SCALE as u64);
        let twice = self.add(ctx, half.cell, half.cell);
        let out = self.add(ctx, k_ln2.cell, twice);
        self.bound(ctx, out)
    }

    /// Canonical sinh/cosh quotient. In particular this does not replace it
    /// with the differently rounded (1-exp(-2|x|))/(1+exp(-2|x|)).
    pub fn canonical_tanh(&self, ctx: &mut Ctx, x: I64) -> I64 {
        let up = self.canonical_exp(ctx, x);
        let negated = self.gate.neg(ctx, x.cell);
        let negated = self.bound(ctx, negated);
        let down = self.canonical_exp(ctx, negated);
        let difference = self.sub(ctx, up.cell, down.cell);
        let sum = self.add(ctx, up.cell, down.cell);
        let (sinh, _) = self.floor_div_const(ctx, difference, 2);
        let (cosh, _) = self.floor_div_const(ctx, sum, 2);
        self.div(ctx, sinh.cell, cosh)
    }

    pub fn canonical_sin(&self, ctx: &mut Ctx, x: I64) -> Cell {
        let negative = self.is_neg(ctx, x);
        let opposite = self.gate.neg(ctx, x.cell);
        let magnitude = self.gate.select(ctx, opposite, x.cell, negative);
        self.canonical_sin_magnitude(ctx, negative, magnitude)
    }

    // cos shifts an int64 argument by pi/2; the shifted value can cross the
    // int64 boundary while its magnitude still fits the compiler's 64-bit
    // trigonometric reduction domain. A 65-bit signed window preserves it.
    fn canonical_sin_value(&self, ctx: &mut Ctx, x: Cell) -> Cell {
        let offset = self.add(ctx, x, Constant(fe(1i128 << 64)));
        self.range.range_check(ctx, offset, 65);
        let negative = self.range.is_less_than(ctx, offset, Constant(fe(1i128 << 64)), 65);
        let opposite = self.gate.neg(ctx, x);
        let magnitude = self.gate.select(ctx, opposite, x, negative);
        self.range.range_check(ctx, magnitude, 64);
        self.canonical_sin_magnitude(ctx, negative, magnitude)
    }

    fn canonical_sin_magnitude(&self, ctx: &mut Ctx, negative: Cell, magnitude: Cell) -> Cell {
        let (_, r) = self.floor_div_const(ctx, magnitude, 26986075410);
        let below_pi = self.range.is_less_than(ctx, r, Constant(fe(13493037705)), 35);
        let folded = self.gate.not(ctx, below_pi);
        let reduced = self.gate.mul_add(ctx, folded, Constant(fe(-13493037705)), r);
        let mirrored = self.range.is_less_than(ctx, Constant(fe(6746518852)), reduced, 34);
        let reflection = self.sub(ctx, Constant(fe(13493037705)), reduced);
        let s = self.gate.select(ctx, reflection, reduced, mirrored);
        let squared = self.mul(ctx, s, s);
        let polynomial = self.canonical_horner(ctx, squared.cell, &SIN_Q);
        let output = self.mul(ctx, s, polynomial);
        let flip = self.gate.xor(ctx, negative, folded);
        let opposite = self.gate.neg(ctx, output.cell);
        self.gate.select(ctx, opposite, output.cell, flip)
    }

    pub fn canonical_cos(&self, ctx: &mut Ctx, x: Cell) -> Cell {
        let shifted = self.add(ctx, x, Constant(fe(6746518852)));
        self.canonical_sin_value(ctx, shifted)
    }

    pub fn canonical_arctan2(&self, ctx: &mut Ctx, y: I64, x: I64) -> Cell {
        let negative_y = self.is_neg(ctx, y);
        let negative_x = self.is_neg(ctx, x);
        let minus_y = self.gate.neg(ctx, y.cell);
        let minus_x = self.gate.neg(ctx, x.cell);
        let ay = self.gate.select(ctx, minus_y, y.cell, negative_y);
        let ax = self.gate.select(ctx, minus_x, x.cell, negative_x);
        let angle = self.canonical_arctan2_nonneg(ctx, ay, ax);
        let reflection = self.sub(ctx, Constant(fe(13493037705)), angle);
        let angle = self.gate.select(ctx, reflection, angle, negative_x);
        let opposite = self.gate.neg(ctx, angle);
        self.gate.select(ctx, opposite, angle, negative_y)
    }

    pub fn canonical_arctan2_nonneg(&self, ctx: &mut Ctx, y: Cell, x: Cell) -> Cell {
        let swap = self.range.is_less_than(ctx, x, y, 64);
        let smaller = self.gate.select(ctx, x, y, swap);
        let larger = self.gate.select(ctx, y, x, swap);
        let zero = self.gate.is_zero(ctx, larger);
        let divisor = self.add(ctx, larger, zero);
        let ratio = self.div_by_abs(ctx, smaller, divisor, divisor);
        let squared = self.mul(ctx, ratio.cell, ratio.cell);
        let polynomial = self.canonical_horner(ctx, squared.cell, &ATAN_Q);
        let angle = self.mul(ctx, ratio.cell, polynomial);
        let reflection = self.sub(ctx, Constant(fe(6746518852)), angle.cell);
        self.gate.select(ctx, reflection, angle.cell, swap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
    use halo2_base::halo2_proofs::dev::MockProver;

    #[test]
    fn frozen_q32_vectors() {
        let vectors: serde_json::Value = serde_json::from_str(include_str!(
            "../../../hw_check/tests/data/canonical_q32.json")).unwrap();
        let mut builder = BaseCircuitBuilder::<Fr>::from_stage(CircuitBuilderStage::Mock);
        builder.set_k(15);
        builder.set_lookup_bits(8);
        builder.set_instance_columns(1);
        let q = Q32::new(&builder);
        let ctx = builder.main(0);
        for vector in vectors.as_array().unwrap() {
            let cell = ctx.load_witness(fe(vector["x"].as_i64().unwrap() as i128));
            let x = q.bound(ctx, cell);
            let output = match vector["op"].as_str().unwrap() {
                "exp" => q.canonical_exp(ctx, x).cell,
                "log" => q.canonical_log(ctx, x).cell,
                "tanh" => q.canonical_tanh(ctx, x).cell,
                "sin" => q.canonical_sin(ctx, x),
                "cos" => q.canonical_cos(ctx, x.cell),
                "arctan2" => {
                    let y = ctx.load_witness(fe(vector["y"].as_i64().unwrap() as i128));
                    let y = q.bound(ctx, y);
                    q.canonical_arctan2(ctx, y, x)
                },
                op => panic!("unknown primitive {op}"),
            };
            let expected = vector["expected"].as_i64().unwrap() as i128;
            assert_eq!(signed(output.value()), expected, "{vector}");
            q.gate.assert_is_const(ctx, &output, &fe(expected));
        }
        builder.calculate_params(Some(20));
        MockProver::run(15, &builder, vec![vec![]]).unwrap().assert_satisfied();
        builder.clear();
    }
}
