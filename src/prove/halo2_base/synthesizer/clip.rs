//! Constant clamp with one disjoint-region range check.
use super::plan::Clip;
use super::*;

impl Halo2BaseSynthesizer<'_> {
    pub(super) fn constant_clip(&mut self, clip: Clip) -> Result<Halo2BaseCell, ProvingError> {
        let x = self.cell_of_vid.get(&clip.input).copied()
            .ok_or_else(|| ProvingError::synthesis("clip input without a cell"))?;
        self.ensure_signed64(&x)?;
        let (lo, hi) = (clip.lo as i128, clip.hi as i128);
        let value = kernel::fp_to_i128(x.value).unwrap_or(0);
        let low = self.hint(Fp::from((value < lo) as u64));
        let high = self.hint(Fp::from((value > hi) as u64));
        self.assert_bit(&low)?;
        self.assert_bit(&high)?;
        let both = self.mul_i(&low, &high)?;
        let zero = self.const_cell(Fp::zero());
        self.constrain_equal(&both, &zero);
        // With D=hi-lo and K=floor(2^64/(D+1))+1, constrain the selected
        // region expression into [0,2^64): lo-1-x for low, x-hi-1 for
        // high, and D*K-(K-1)*(x-lo) for middle. For signed64 x and bounds,
        // each expression lies in that window exactly in its own region.
        // Outside the middle region its distance is >=2^64. All products
        // are below 2^129, so no field wrap can fake region membership.
        let d = hi - lo;
        let k = (1i128 << 64) / (d + 1) + 1;
        let fe = kernel::i128_to_fp;
        let middle = fe(d) * fe(k) + fe(lo) * fe(k - 1);
        let linear = |s: &mut Self, coefficient: Fp, constant: Fp| {
            s.gate_cell(x.value * coefficient + constant, |gate, ctx|
                gate.mul_add(ctx, x.quantum(), QuantumCell::Constant(coefficient), QuantumCell::Constant(constant)))
        };
        let a = linear(self, -fe(k - 1), middle);
        let b = linear(self, fe(k - 2), fe(lo - 1) - middle);
        let c = linear(self, fe(k), -fe(hi + 1) - middle);
        let u = self.multiply_add(&a, &high, &c);
        let u = self.multiply_add(&u, &low, &b);
        self.range_check(&u, 64);
        let to_lo = self.constant_minus(lo, &x);
        let to_hi = self.constant_minus(hi, &x);
        let result = self.multiply_add(&x, &high, &to_hi);
        let result = self.multiply_add(&result, &low, &to_lo);
        self.prove_range(&result, Interval { lo, hi });
        Ok(result)
    }
}
