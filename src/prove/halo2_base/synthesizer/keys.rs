//! `IR::KeyLess`: the lexicographic order of sort keys `(value, index)` by
//! one comparison of packed keys.

use halo2_base::gates::GateInstructions;
use halo2_base::QuantumCell;

use super::ranges::{Interval, Operand};
use super::{Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel;

impl Halo2BaseSynthesizer<'_> {
    /// `[(v1, i1) < (v2, i2)]` lexicographically.
    ///
    /// Each key is packed as `K = v·2^b + i` (`b = index_bits`, one gate,
    /// built once per pair). The values are constrained into the signed
    /// 64-bit window, as the operands of any comparison are, and the indices
    /// into `[0, 2^b)` (range-checked unless already proven, e.g. by the
    /// bounds assertion of a hinted index or as constants). Then
    /// `K1 - K2 = (v1 - v2)·2^b + (i1 - i2)` with `|i1 - i2| < 2^b`: when the
    /// values differ the value term dominates, when they are equal the index
    /// term decides, so the sign of `K1 - K2` is the lexicographic order.
    /// `|K1 - K2| < 2^(64+b)`, and `less_than` pins its answer with one range
    /// check sized by the keys' proven ranges (its own argument), far below
    /// `p / 2`. One comparison replaces the three comparisons and two logic
    /// gates of the composed form.
    pub(super) fn packed_key_less(
        &mut self,
        v1: &Halo2BaseCell,
        i1: &Halo2BaseCell,
        v2: &Halo2BaseCell,
        i2: &Halo2BaseCell,
        index_bits: u32,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let k1 = self.packed_key(v1, i1, index_bits)?;
        let k2 = self.packed_key(v2, i2, index_bits)?;
        self.less_than(Operand::Cell(k1), Operand::Cell(k2), None)
    }

    /// `v·2^b + i`, with `v` in the window and `i` in `[0, 2^b)`, and the
    /// range that follows from those.
    fn packed_key(&mut self, v: &Halo2BaseCell, i: &Halo2BaseCell, index_bits: u32) -> Result<Halo2BaseCell, ProvingError> {
        if let Some(key) = self.packed_keys.get(&(v.id, i.id, index_bits)) {
            return Ok(*key);
        }
        let indices = Interval::bits(index_bits as usize)
            .filter(|_| index_bits <= 32)
            .ok_or_else(|| ProvingError::synthesis("key comparison: index width above 32 bits"))?;
        self.ensure_signed64(v)?;
        if !self.proven_range(i).is_some_and(|r| r.within(&indices)) {
            self.range_check(i, index_bits as usize);
        }
        let scale = 1i128 << index_bits;
        let scale_fp = kernel::i128_to_fp(scale);
        let key = self.gate_cell(v.value * scale_fp + i.value, |gate, ctx| {
            gate.mul_add(ctx, v.assigned, QuantumCell::Constant(scale_fp), i.assigned)
        });
        let (vr, ir) = (Interval::SIGNED64, indices);
        let range = Interval::new(vr.lo * scale + ir.lo, vr.hi * scale + ir.hi)
            .ok_or_else(|| ProvingError::synthesis("key comparison: key range overflow"))?;
        self.prove_range(&key, range);
        self.packed_keys.insert((v.id, i.id, index_bits), key);
        Ok(key)
    }
}
