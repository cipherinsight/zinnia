//! `IR::KeyLess`: the lexicographic order of sort keys `(value, index)` by
//! one comparison of packed keys.

use halo2_base::gates::GateInstructions;
use halo2_base::QuantumCell;

use super::ranges::{Interval, Operand};
use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::prove::traits::Synthesizer;

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
    pub(super) fn key_less_by_packing(
        &mut self,
        v1: &Halo2BaseCell,
        i1: &Halo2BaseCell,
        v2: &Halo2BaseCell,
        i2: &Halo2BaseCell,
        index_bits: u32,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let k1 = self.packed_key(v1, i1, index_bits)?;
        let k2 = self.packed_key(v2, i2, index_bits)?;
        self.less_of_packed_keys(&k1, &k2)
    }

    /// `[K1 < K2]` for packed keys (`IR::PackedKeyLess`, and the second half
    /// of `IR::KeyLess`): `less_than` on the keys' proven ranges, which
    /// `packed_key` establishes and selections among keys keep (a select's
    /// output is one of its branches), or one gap range check when asserted.
    pub(super) fn less_of_packed_keys(
        &mut self,
        k1: &Halo2BaseCell,
        k2: &Halo2BaseCell,
    ) -> Result<Halo2BaseCell, ProvingError> {
        if self.is_asserted_comparison() {
            return self.assert_packed_less(k1, k2);
        }
        self.less_than(Operand::Cell(*k1), Operand::Cell(*k2), None)
    }

    /// The index `i` of a packed key `K` (`IR::PackedKeyIndex`).
    ///
    /// The prover supplies `q` and `i`; the circuit range-checks `i` to
    /// `index_bits` bits and `q` into the signed 64-bit window and asserts
    /// `q·2^b + i == K`.
    ///
    /// Soundness. `K` is a packed key (`packed_key`) or a selection among
    /// packed keys, so in every accepted witness `K = v·2^b + j` with `v` in
    /// the window and `j` in `[0, 2^b)`, as integers, and `|K| < 2^(63+b)`.
    /// The checks admit `q·2^b + i` only in `[-2^(63+b), 2^(63+b))` with the
    /// same bounds on its parts; on that range the map `(q, i) ↦ q·2^b + i`
    /// is injective and below `p / 2` in magnitude, so the field equation
    /// is an integer equation and `(q, i) = (v, j)` is its only solution.
    /// The key's proven range (required) is what guarantees the packing.
    pub(super) fn index_of_packed_key(
        &mut self,
        key: &Halo2BaseCell,
        index_bits: u32,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let indices = Interval::bits(index_bits as usize)
            .filter(|_| index_bits <= 32)
            .ok_or_else(|| ProvingError::synthesis("packed key: index width above 32 bits"))?;
        let scale = 1i128 << index_bits;
        let keys = Interval::new(Interval::SIGNED64.lo * scale, Interval::SIGNED64.hi * scale + scale - 1)
            .ok_or_else(|| ProvingError::synthesis("packed key: key range overflow"))?;
        if !self.proven_range(key).is_some_and(|r| r.within(&keys)) {
            return Err(ProvingError::synthesis("packed key index: operand is not a packed key"));
        }
        let (q_value, i_value) = kernel::packed_key_parts(key.value, index_bits);
        let q = self.hint(q_value);
        let i = self.hint(i_value);
        self.range_check(&i, index_bits as usize);
        self.prove_range(&i, indices);
        self.ensure_signed64(&q)?;
        let scale_fp = kernel::i128_to_fp(scale);
        let rebuilt = self.gate_cell(q.value * scale_fp + i.value, |gate, ctx| {
            gate.mul_add(ctx, q.assigned, QuantumCell::Constant(scale_fp), i.assigned)
        });
        if rebuilt.value != key.value {
            self.note_unsat(|| "Witness does not satisfy circuit constraints: a packed key's parts".to_string());
        }
        self.constrain_equal(&rebuilt, key);
        Ok(i)
    }

    /// `K1 < K2` asserted (the comparison's only use is `Assert`, see
    /// `plan.rs`): no result bit, one range check of the gap `K2 - K1 - 1`
    /// at a width above the largest gap the keys' proven ranges allow. A
    /// false order makes the gap negative, a field element near `p`, which
    /// fails the check.
    fn assert_packed_less(&mut self, k1: &Halo2BaseCell, k2: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let ranges = self.proven_range(k1).zip(self.proven_range(k2));
        let gap_range = ranges
            .and_then(|(r1, r2)| r2.sub(&r1))
            .and_then(|g| g.offset(-1))
            .ok_or_else(|| ProvingError::synthesis("key comparison: keys without proven ranges"))?;
        let gap_value = k2.value - k1.value - Fp::one();
        if kernel::fp_to_i128(gap_value).is_none_or(|g| g < 0) {
            self.note_unsat(|| "Witness does not satisfy circuit constraints: an asserted key order is false".to_string());
        }
        if gap_range.lo < 0 {
            let difference = self.difference(k2, k1);
            let gap = self.add_constant(&difference, -1);
            let bits = (u128::BITS - (gap_range.hi.max(1) as u128).leading_zeros()) as usize;
            let width = self.lookup_aligned(bits);
            self.range_check(&gap, width);
        }
        self.constant_bool(true)
    }

    /// `v·2^b + i`, with `v` in the window and `i` in `[0, 2^b)`, and the
    /// range that follows from those.
    pub(super) fn packed_key(&mut self, v: &Halo2BaseCell, i: &Halo2BaseCell, index_bits: u32) -> Result<Halo2BaseCell, ProvingError> {
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
