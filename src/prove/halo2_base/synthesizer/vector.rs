//! Keep bitwise dataflow in constrained Boolean form across shifts and selects.
//! Scalar reconstruction is delayed only across consumers selected by the plan.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum BitOp { And, Or, Xor }

impl Halo2BaseSynthesizer<'_> {
    fn vector_input(&self, x: &Halo2BaseCell) -> bool {
        self.proven_range(x).is_some_and(|r| r.within(&Interval::SIGNED64))
    }

    pub(super) fn materialize_vector(&mut self, x: &Halo2BaseCell) -> Option<Halo2BaseCell> {
        if !self.vector_deferred.contains(&x.id) { return None; }
        let bits = self.bit_cache.get(&x.id)?.clone();
        Some(self.pack_vector(&bits))
    }

    fn pack_vector(&mut self, bits: &[Halo2BaseCell]) -> Halo2BaseCell {
        let end = bits.iter().rposition(|bit| !self.is_const_cell(bit) || bit.value != Fp::zero()).map_or(0, |i| i + 1);
        if end == 0 { return self.const_cell(Fp::zero()); }
        if end == 1 { return bits[0]; }
        let bits = &bits[..end];
        let weights: Vec<_> = (0..end).map(|i| kernel::i128_to_fp(if i == 63 { -(1i128 << 63) } else { 1i128 << i })).collect();
        let value = bits.iter().zip(&weights).fold(Fp::zero(), |acc, (b, w)| acc + b.value * w);
        let out = self.gate_cell(value, |gate, ctx| gate.inner_product(ctx,
            bits.iter().map(|b| b.quantum()), weights.iter().map(|w| QuantumCell::Constant(*w))));
        self.bit_cache.insert(out.id, bits.to_vec());
        self.prove_range(&out, if end == 64 { Interval::SIGNED64 } else { Interval { lo: 0, hi: (1i128 << end) - 1 } });
        out
    }

    fn vector_result(&mut self, bits: Vec<Halo2BaseCell>) -> Halo2BaseCell {
        // Even a deferred one-bit integer is already its bit cell.
        let end = bits.iter().rposition(|bit| !self.is_const_cell(bit) || bit.value != Fp::zero()).map_or(0, |i| i + 1);
        let out = if end <= 1 || !self.plan.vector_links.contains(&self.current_stmt) {
            self.pack_vector(&bits)
        } else {
            let value = bits.iter().enumerate().fold(Fp::zero(), |acc, (i, bit)| acc + bit.value
                * kernel::i128_to_fp(if i == 63 { -(1i128 << 63) } else { 1i128 << i }));
            let out = self.cellless(value);
            self.vector_deferred.insert(out.id);
            out
        };
        self.prove_range(&out, if end == 64 { Interval::SIGNED64 } else { Interval { lo: 0, hi: (1i128 << end) - 1 } });
        self.bit_cache.insert(out.id, bits);
        out
    }

    fn xor_bit(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        if a.id == b.id { return self.constant_bool(false); }
        for (constant, other) in [(a, b), (b, a)] {
            if let Some(truth) = self.constant_truth(constant) {
                return if truth { self.logical_not(other) } else { Ok(*other) };
            }
        }
        let out = self.gate_cell(a.value + b.value - Fp::from(2u64) * a.value * b.value,
            |gate, ctx| gate.xor(ctx, a.quantum(), b.quantum()));
        self.mark_bit(&out);
        Ok(out)
    }

    pub(super) fn vector_binary(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell, op: BitOp)
        -> Result<Option<Halo2BaseCell>, ProvingError> {
        if !self.plan.vector_ops.contains(&self.current_stmt) || !self.vector_input(a) || !self.vector_input(b) { return Ok(None); }
        // A constant positive mask only needs the corresponding low bits of
        // its other operand; input bytes must not be expanded to 64 bits.
        let mask_width = if matches!(op, BitOp::And) {
            [(a, b), (b, a)].iter().find_map(|(c, _)| self.is_const_cell(c)
                .then(|| kernel::fp_to_i128(c.value)).flatten().filter(|v| *v >= 0)
                .map(|v| (128 - v.leading_zeros()) as usize)).unwrap_or(64)
        } else { 64 };
        if mask_width == 0 { return Ok(Some(self.const_cell(Fp::zero()))); }
        let a_bits = self.low_bits(a, mask_width)?; let b_bits = self.low_bits(b, mask_width)?;
        let mut bits = Vec::with_capacity(64);
        for (x, y) in a_bits.iter().zip(&b_bits) {
            bits.push(match op { BitOp::And => self.logical_and(x, y)?, BitOp::Or => self.logical_or(x, y)?, BitOp::Xor => self.xor_bit(x, y)? });
        }
        let zero = self.constant_bool(false)?;
        bits.resize(64, zero);
        Ok(Some(self.vector_result(bits)))
    }

    pub(super) fn vector_not(&mut self, a: &Halo2BaseCell) -> Result<Option<Halo2BaseCell>, ProvingError> {
        if !self.plan.vector_ops.contains(&self.current_stmt) || !self.vector_input(a) { return Ok(None); }
        let mut bits = self.low_bits(a, 64)?;
        for bit in &mut bits { *bit = self.logical_not(bit)?; }
        Ok(Some(self.vector_result(bits)))
    }

    pub(super) fn vector_shift(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell, left: bool)
        -> Result<Option<Halo2BaseCell>, ProvingError> {
        if !self.plan.vector_ops.contains(&self.current_stmt) || !self.vector_input(a) || !self.is_const_cell(b) { return Ok(None); }
        let signed_count = kernel::fp_to_i64(b.value);
        if kernel::i64_to_fp(signed_count) != b.value { return Ok(None); }
        let count = kernel::int64_shift_count(signed_count);
        let original = self.low_bits(a, 64)?;
        let zero = self.constant_bool(false)?;
        let bits = if left {
            match count { Some(c) => (0..64).map(|i| if i >= c as usize { original[i - c as usize] } else { zero }).collect(), None => vec![zero; 64] }
        } else {
            let c = count.unwrap_or(63) as usize;
            (0..64).map(|i| original[(i + c).min(63)]).collect()
        };
        Ok(Some(self.vector_result(bits)))
    }

    pub(super) fn vector_select(&mut self, condition: &Halo2BaseCell, t: &Halo2BaseCell, f: &Halo2BaseCell)
        -> Result<Option<Halo2BaseCell>, ProvingError> {
        if !self.plan.vector_ops.contains(&self.current_stmt) || !self.vector_input(t) || !self.vector_input(f) { return Ok(None); }
        let t_bits = self.low_bits(t, 64)?; let f_bits = self.low_bits(f, 64)?;
        let mut bits = Vec::with_capacity(64);
        for (t, f) in t_bits.iter().zip(&f_bits) {
            bits.push(if t.id == f.id { *t } else { self.select_bit(condition, t, f)? });
        }
        Ok(Some(self.vector_result(bits)))
    }
}
