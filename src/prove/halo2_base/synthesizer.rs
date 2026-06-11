use std::collections::{HashMap, HashSet};

use halo2_base::{
    gates::{
        circuit::builder::BaseCircuitBuilder,
        flex_gate::{GateChip, GateInstructions},
        range::RangeInstructions,
    },
    halo2_proofs::halo2curves::bn256::Fr as Fp,
    AssignedValue, QuantumCell,
};

use crate::circuit_input::{InputPath, ResolvedWitness};
use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;

#[derive(Clone, Copy, Debug)]
pub struct Halo2BaseCell {
    pub id: u64,
    pub assigned: AssignedValue<Fp>,
    pub value: Fp,
}

pub struct Halo2BaseSynthesizer<'a> {
    builder: &'a mut BaseCircuitBuilder<Fp>,
    witness: Option<ResolvedWitness>,
    params: ProvingParams,
    public_values: Vec<Fp>,
    memories: HashMap<u32, Vec<Fp>>,
    memory_cells: HashMap<u32, Vec<Option<Halo2BaseCell>>>,
    memory_init: HashMap<u32, Fp>,
    dynamic_writes: HashMap<u32, Vec<(Halo2BaseCell, Halo2BaseCell)>>,
    dyn_meta_bounds: HashMap<u32, (u32, Vec<u32>)>,
    next_id: u64,
    bit_cells: HashSet<halo2_base::ContextCell>,
    bit_cell_ids: HashSet<u64>,
    const_cell_ids: HashSet<u64>,
    select_many_cache: HashMap<Vec<u64>, Vec<Halo2BaseCell>>,
    u128_lt_bit_cache: HashMap<(u64, u64, u32), Vec<Halo2BaseCell>>,
    // Affine address lineage: cell id -> sum of dynamic terms plus offset,
    // enforced by the gates that built the cell. Lets dynamic-address memory
    // reads mux over only the slots reachable from the roots instead of the
    // whole segment.
    addr_lineage: HashMap<u64, AffineLin>,
    // eq(root, v) indicator bits, shared across all reads on the same root.
    indicator_cache: HashMap<(u64, i64), Halo2BaseCell>,
    // (root id, enumerated value set) one-hot memberships already asserted.
    membership_asserted: HashSet<(u64, Vec<i64>)>,
    // Stage-1 gathered planes for factored two-root reads, keyed by
    // (segment, write version, root id, root scale, residual offset).
    plane_cache: HashMap<(u32, u64, u64, i64, i64), Halo2BaseCell>,
    // Bumped on every write to a segment; invalidates plane_cache entries.
    write_version: HashMap<u32, u64>,
    // Cells already constrained to the signed-64 window (x + 2^63 < 2^64)
    // resp. the unsigned-64 window (x < 2^64). Comparisons skip their input
    // range checks for cells whose window is already enforced; selects
    // propagate the facts (the output provably equals one branch).
    signed64_cells: HashSet<u64>,
    unsigned64_cells: HashSet<u64>,
    // eq(write addr, read addr) bits for dynamic-write overlays. The read
    // side is keyed by cell id for dynamic addresses and by value for
    // constant addresses (constants with equal values are interchangeable).
    overlay_eq_cache: HashMap<(u64, OverlayReadKey), Halo2BaseCell>,
    // Constraint-derived value intervals (signed i64 view) for cells the
    // provenance table can't see (synthesizer-internal helpers). IR-level
    // cells read the table live via `vid_of_cell`. Witness-independent by
    // construction.
    value_interval: HashMap<u64, (i64, i64)>,
    // ValueId ↔ cell correspondence, maintained by `on_stmt_result`, for
    // consuming the structural-provenance table (affine views and
    // intervals are derived ONCE in the IRBuilder; see src/provenance.rs).
    prov: &'a crate::provenance::ProvenanceTable,
    vid_of_cell: HashMap<u64, crate::types::ValueId>,
    cell_of_vid: HashMap<crate::types::ValueId, Halo2BaseCell>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum OverlayReadKey {
    Cell(u64),
    Const(i64),
}

/// Affine view of an integer cell: `cell == Σ root_i * scale_i + offset`,
/// with at most two dynamic roots and non-zero scales.
#[derive(Clone, Debug, Default)]
struct AffineLin {
    terms: Vec<(Halo2BaseCell, i64)>,
    offset: i64,
}


impl<'a> Halo2BaseSynthesizer<'a> {
    pub fn new(
        builder: &'a mut BaseCircuitBuilder<Fp>,
        witness: Option<ResolvedWitness>,
        params: ProvingParams,
        provenance: &'a crate::provenance::ProvenanceTable,
    ) -> Self {
        Self {
            builder,
            witness,
            params,
            public_values: Vec::new(),
            memories: HashMap::new(),
            memory_cells: HashMap::new(),
            memory_init: HashMap::new(),
            dynamic_writes: HashMap::new(),
            dyn_meta_bounds: HashMap::new(),
            next_id: 0,
            bit_cells: HashSet::new(),
            bit_cell_ids: HashSet::new(),
            const_cell_ids: HashSet::new(),
            select_many_cache: HashMap::new(),
            u128_lt_bit_cache: HashMap::new(),
            addr_lineage: HashMap::new(),
            indicator_cache: HashMap::new(),
            membership_asserted: HashSet::new(),
            plane_cache: HashMap::new(),
            write_version: HashMap::new(),
            signed64_cells: HashSet::new(),
            unsigned64_cells: HashSet::new(),
            overlay_eq_cache: HashMap::new(),
            value_interval: HashMap::new(),
            prov: provenance,
            vid_of_cell: HashMap::new(),
            cell_of_vid: HashMap::new(),
        }
    }

    pub fn public_values(&self) -> &[Fp] {
        &self.public_values
    }

    fn ctx(&mut self) -> &mut halo2_base::Context<Fp> {
        self.builder.main(0)
    }

    fn cell(&mut self, value: Fp) -> Halo2BaseCell {
        let assigned = self.ctx().load_witness(value);
        self.wrap(assigned, value)
    }

    fn const_cell(&mut self, value: Fp) -> Halo2BaseCell {
        let assigned = self.ctx().load_constant(value);
        let cell = self.wrap(assigned, value);
        self.const_cell_ids.insert(cell.id);
        let v = kernel::fp_to_i64(value);
        if kernel::i64_to_fp(v) == value {
            self.value_interval.insert(cell.id, (v, v));
        }
        cell
    }

    /// Constraint-derived signed-i64 interval of a cell, if known. IR-level
    /// cells read the provenance table LIVE (post-hoc tightens from later
    /// asserts are visible); internal cells fall back to the local map.
    fn interval_of(&self, cell: &Halo2BaseCell) -> Option<(i64, i64)> {
        if let Some(vid) = self.vid_of_cell.get(&cell.id) {
            if let Some(iv) = self.prov.interval_of(*vid) {
                return Some((iv.lo, iv.hi));
            }
        }
        if let Some(iv) = self.value_interval.get(&cell.id) {
            return Some(*iv);
        }
        if self.bit_cell_ids.contains(&cell.id) {
            return Some((0, 1));
        }
        None
    }

    fn wrap(&mut self, assigned: AssignedValue<Fp>, value: Fp) -> Halo2BaseCell {
        let id = self.next_id;
        self.next_id += 1;
        Halo2BaseCell { id, assigned, value }
    }

    fn mark_bit(&mut self, cell: &Halo2BaseCell) {
        self.bit_cell_ids.insert(cell.id);
        if let Some(ptr) = cell.assigned.cell {
            self.bit_cells.insert(ptr);
        }
    }

    fn is_bit(&self, cell: &Halo2BaseCell) -> bool {
        if self.bit_cell_ids.contains(&cell.id) {
            return true;
        }
        cell.assigned
            .cell
            .map(|ptr| self.bit_cells.contains(&ptr))
            .unwrap_or(false)
            || cell.value == Fp::zero()
            || cell.value == Fp::one()
    }

    fn is_const_cell(&self, cell: &Halo2BaseCell) -> bool {
        self.const_cell_ids.contains(&cell.id)
    }

    /// Affine view of `cell`, read live from the provenance table (terms
    /// translate root ValueIds to their cells; a root without a cell yet
    /// degrades to the identity view). Falls back to the identity view
    /// (the cell is its own root) for dynamic cells, and to a pure offset
    /// for constants.
    fn affine_lin_of(&self, cell: &Halo2BaseCell) -> AffineLin {
        if let Some(vid) = self.vid_of_cell.get(&cell.id) {
            if let Some(av) = self.prov.affine_of(*vid) {
                if !av.terms.is_empty() {
                    let mut terms = Vec::with_capacity(av.terms.len());
                    let mut ok = true;
                    for (rvid, scale) in &av.terms {
                        match self.cell_of_vid.get(rvid) {
                            Some(rc) => terms.push((*rc, *scale)),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        return AffineLin {
                            terms,
                            offset: av.offset,
                        };
                    }
                }
            }
        }
        if let Some(lin) = self.addr_lineage.get(&cell.id) {
            return lin.clone();
        }
        if self.is_const_cell(cell) {
            return AffineLin { terms: Vec::new(), offset: kernel::fp_to_i64(cell.value) };
        }
        AffineLin { terms: vec![(*cell, 1)], offset: 0 }
    }

    /// eq(root, v) indicator bit, shared across reads on the same root.
    fn root_indicator(&mut self, root: &Halo2BaseCell, v: i64) -> Result<Halo2BaseCell, ProvingError> {
        if let Some(ind) = self.indicator_cache.get(&(root.id, v)) {
            return Ok(*ind);
        }
        let v_cell = self.const_cell(kernel::i64_to_fp(v));
        let ind = self.eq_i(&root.clone(), &v_cell)?;
        self.indicator_cache.insert((root.id, v), ind);
        Ok(ind)
    }

    /// Assert (once per root and value set) that exactly one of the
    /// eq(root, v) indicators is high, pinning the root onto the set.
    fn assert_one_hot(&mut self, root: &Halo2BaseCell, vs: &[i64]) -> Result<(), ProvingError> {
        let key = (root.id, vs.to_vec());
        if self.membership_asserted.contains(&key) {
            return Ok(());
        }
        let mut sum: Option<Halo2BaseCell> = None;
        for v in vs {
            let ind = self.root_indicator(root, *v)?;
            sum = Some(match sum {
                Some(acc) => self.add_i(&acc, &ind)?,
                None => ind,
            });
        }
        if let Some(sum) = sum {
            self.assert_const(&sum, Fp::one());
        }
        self.membership_asserted.insert(key);
        Ok(())
    }

    /// Stage-1 gathered plane for factored two-root reads:
    /// `Σ_v eq(root, v) * mem[v * scale + t]` over in-bounds v, cached per
    /// (segment, write version, root, scale, t). The cached value is only
    /// meaningful when the read's full address is separately range-checked.
    #[allow(clippy::too_many_arguments)]
    fn gathered_plane(
        &mut self,
        segment_id: u32,
        root: &Halo2BaseCell,
        scale: i64,
        t: i64,
        len: usize,
        init: Fp,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let version = self.write_version.get(&segment_id).copied().unwrap_or(0);
        let key = (segment_id, version, root.id, scale, t);
        if let Some(plane) = self.plane_cache.get(&key) {
            return Ok(*plane);
        }
        let mut out = self.cell(init);
        for idx in 0..len as i64 {
            let num = idx - t;
            if num % scale != 0 {
                continue;
            }
            let ind = self.root_indicator(root, num / scale)?;
            let slot = self
                .memory_cells
                .get(&segment_id)
                .and_then(|cells| cells.get(idx as usize))
                .copied()
                .flatten()
                .unwrap_or_else(|| self.const_cell(init));
            out = self.select_bit(&ind, &slot, &out)?;
        }
        self.plane_cache.insert(key, out);
        Ok(out)
    }

    /// eq(write addr, read addr) bit for dynamic-write overlays, cached so
    /// repeated reads against the same pending write reuse the constraint.
    fn overlay_hit(
        &mut self,
        write_addr: &Halo2BaseCell,
        read_addr: &Halo2BaseCell,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let key = if self.is_const_cell(read_addr) {
            OverlayReadKey::Const(kernel::fp_to_i64(read_addr.value))
        } else {
            OverlayReadKey::Cell(read_addr.id)
        };
        if let Some(hit) = self.overlay_eq_cache.get(&(write_addr.id, key)) {
            return Ok(*hit);
        }
        let hit = self.eq_i(&write_addr.clone(), &read_addr.clone())?;
        self.overlay_eq_cache.insert((write_addr.id, key), hit);
        Ok(hit)
    }

    fn gate_cell(
        &mut self,
        value: Fp,
        f: impl FnOnce(&GateChip<Fp>, &mut halo2_base::Context<Fp>) -> AssignedValue<Fp>,
    ) -> Halo2BaseCell {
        let gate = GateChip::<Fp>::default();
        let assigned = f(&gate, self.ctx());
        self.wrap(assigned, value)
    }

    fn truthy_bit(&mut self, a: &Halo2BaseCell) -> Halo2BaseCell {
        if self.is_bit(a) {
            return *a;
        }
        let gate = GateChip::<Fp>::default();
        let assigned = gate.is_zero(self.ctx(), a.assigned);
        let is_zero = if a.value == Fp::zero() { Fp::one() } else { Fp::zero() };
        let iz = self.wrap(assigned, is_zero);
        self.mark_bit(&iz);
        let out = self.gate_cell(Fp::one() - iz.value, |gate, ctx| {
            gate.not(ctx, iz.assigned)
        });
        self.mark_bit(&out);
        out
    }

    fn constrain_equal(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) {
        self.ctx().constrain_equal(&a.assigned, &b.assigned);
    }

    fn assert_const(&mut self, a: &Halo2BaseCell, c: Fp) {
        GateChip::<Fp>::default().assert_is_const(self.ctx(), &a.assigned, &c);
    }

    fn unsupported<T>(&self, op: &str) -> Result<T, ProvingError> {
        Err(ProvingError::synthesis(format!(
            "halo2-base backend does not yet support {}",
            op
        )))
    }

    fn get_witness_fp_path(&self, path: &InputPath) -> Option<Fp> {
        self.witness.as_ref().and_then(|w| w.resolve_path(path).ok())
    }

    fn get_external_fp(&self, store_idx: u32, output_idx: u32) -> Option<Fp> {
        self.witness
            .as_ref()
            .and_then(|w| w.resolve_external(store_idx, output_idx).ok())
    }

    fn signed_floor_div_pair(
        &self,
        a: &Halo2BaseCell,
        b: &Halo2BaseCell,
    ) -> Result<(Fp, Fp), ProvingError> {
        let sa = kernel::fp_to_i64(a.value);
        let sb = kernel::fp_to_i64(b.value);
        if sb == 0 {
            if self.witness.is_none() {
                return Ok((Fp::zero(), Fp::zero()));
            }
            return Err(ProvingError::synthesis("division by zero"));
        }
        Ok((kernel::i64_to_fp(sa.div_euclid(sb)), kernel::i64_to_fp(sa.rem_euclid(sb))))
    }

    fn signed_offset_cell(&mut self, a: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let offset = self.const_cell(Fp::from(1u64 << 63));
        self.add_i(a, &offset)
    }

    /// Whether `x + 2^63 < 2^64` is already enforced: previously checked,
    /// a constant (value fixed by the fixed column), or an asserted bit.
    /// Only witness-independent facts qualify — the skip decision must be
    /// identical for every witness or keygen and prove would diverge.
    fn is_signed64_known(&self, x: &Halo2BaseCell) -> bool {
        self.signed64_cells.contains(&x.id)
            || self.bit_cell_ids.contains(&x.id)
            || (self.const_cell_ids.contains(&x.id)
                // The constant's fixed value must actually lie in the
                // signed-64 window (canonical i64 encoding roundtrips).
                && kernel::i64_to_fp(kernel::fp_to_i64(x.value)) == x.value)
    }

    /// Whether `x < 2^64` is already enforced (same qualifying facts).
    fn is_unsigned64_known(&self, x: &Halo2BaseCell) -> bool {
        self.unsigned64_cells.contains(&x.id)
            || self.bit_cell_ids.contains(&x.id)
            || (self.const_cell_ids.contains(&x.id)
                && kernel::fp_to_i64(x.value) >= 0
                && kernel::i64_to_fp(kernel::fp_to_i64(x.value)) == x.value)
    }

    fn signed_lt(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let a_known = self.is_signed64_known(a);
        let b_known = self.is_signed64_known(b);
        let a_shifted = self.signed_offset_cell(a)?;
        let b_shifted = self.signed_offset_cell(b)?;
        let value = if kernel::fp_to_i64(a.value) < kernel::fp_to_i64(b.value) {
            Fp::one()
        } else {
            Fp::zero()
        };
        let range = self.builder.range_chip();
        if !a_known {
            range.range_check(self.ctx(), a_shifted.assigned, 64);
            self.signed64_cells.insert(a.id);
        }
        if !b_known {
            range.range_check(self.ctx(), b_shifted.assigned, 64);
            self.signed64_cells.insert(b.id);
        }
        let assigned = range.is_less_than(self.ctx(), a_shifted.assigned, b_shifted.assigned, 64);
        let out = self.wrap(assigned, value);
        self.mark_bit(&out);
        Ok(out)
    }

    fn quantize_fp(&self, value: f64) -> Fp {
        kernel::quantize_to_fp(value, self.params.precision_bits)
    }

    fn poly_eval_horner(&mut self, x: &Halo2BaseCell, coefs: &[Fp]) -> Result<Halo2BaseCell, ProvingError> {
        if coefs.is_empty() {
            return Err(ProvingError::synthesis("empty polynomial"));
        }
        let mut acc = self.constant_float(0.0)?;
        let first = self.const_cell(coefs[0]);
        acc = self.add_f(&acc, &first)?;
        for coef in coefs.iter().skip(1) {
            let prod = self.mul_f(x, &acc)?;
            let coef_cell = self.const_cell(*coef);
            acc = self.add_f(&prod, &coef_cell)?;
        }
        Ok(acc)
    }

    fn signed_negate_if(&mut self, value: &Halo2BaseCell, sign: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let neg = self.gate_cell(-value.value, |gate, ctx| gate.neg(ctx, value.assigned));
        self.select_bit(sign, &neg, value)
    }

    fn assert_close_f(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell, epsilon: f64) -> Result<(), ProvingError> {
        let diff = self.sub_f(a, b)?;
        let eps_pos = self.constant_float(epsilon)?;
        let eps_neg = self.constant_float(-epsilon)?;
        let lt_pos = self.lt_f(&diff, &eps_pos)?;
        let gt_neg = self.lt_f(&eps_neg, &diff)?;
        self.assert_true(&lt_pos)?;
        self.assert_true(&gt_neg)?;
        Ok(())
    }
}

impl Synthesizer for Halo2BaseSynthesizer<'_> {
    fn on_stmt_result(
        &mut self,
        _stmt_id: u32,
        value_id: crate::types::ValueId,
        cell: &Self::CellRef,
    ) {
        // Maintain the ValueId ↔ cell correspondence for live
        // provenance-table reads (affine views translate root ValueIds to
        // cells; intervals resolve per lookup so post-hoc assert tightens
        // are visible).
        self.vid_of_cell.insert(cell.id, value_id);
        self.cell_of_vid.insert(value_id, *cell);
    }

    type CellRef = Halo2BaseCell;

    fn constant_int(&mut self, value: i64) -> Result<Self::CellRef, ProvingError> {
        Ok(self.const_cell(kernel::i64_to_fp(value)))
    }

    fn constant_float(&mut self, value: f64) -> Result<Self::CellRef, ProvingError> {
        Ok(self.const_cell(kernel::quantize_to_fp(
            value,
            self.params.precision_bits,
        )))
    }

    fn constant_bool(&mut self, value: bool) -> Result<Self::CellRef, ProvingError> {
        let cell = self.const_cell(if value { Fp::one() } else { Fp::zero() });
        self.mark_bit(&cell);
        Ok(cell)
    }

    fn add_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        // Affine lineage now lives in the provenance table (derived once
        // in the IRBuilder); `affine_lin_of` reads it live.
        Ok(self.gate_cell(a.value + b.value, |gate, ctx| gate.add(ctx, a.assigned, b.assigned)))
    }

    fn sub_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Ok(self.gate_cell(a.value - b.value, |gate, ctx| gate.sub(ctx, a.assigned, b.assigned)))
    }

    fn mul_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Ok(self.gate_cell(a.value * b.value, |gate, ctx| gate.mul(ctx, a.assigned, b.assigned)))
    }

    fn div_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let value = b.value.invert().unwrap_or(Fp::zero()) * a.value;
        Ok(self.gate_cell(value, |gate, ctx| gate.div_unsafe(ctx, a.assigned, b.assigned)))
    }

    fn floor_div_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let (q, r) = self.signed_floor_div_pair(a, b)?;
        let q_cell = self.cell(q);
        let r_cell = self.cell(r);
        let recomposed = self.gate_cell(b.value * q + r, |gate, ctx| {
            gate.mul_add(ctx, b.assigned, q_cell.assigned, r_cell.assigned)
        });
        self.constrain_equal(&recomposed, a);
        Ok(q_cell)
    }

    fn mod_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let (q, r) = self.signed_floor_div_pair(a, b)?;
        let q_cell = self.cell(q);
        let r_cell = self.cell(r);
        let recomposed = self.gate_cell(b.value * q + r, |gate, ctx| {
            gate.mul_add(ctx, b.assigned, q_cell.assigned, r_cell.assigned)
        });
        self.constrain_equal(&recomposed, a);
        Ok(r_cell)
    }

    fn pow_i(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let e = kernel::fp_to_i64(exp.value);
        if e < 0 {
            let neg_exp = self.constant_int(-e)?;
            let pos = self.pow_i(base, &neg_exp)?;
            return self.inv_i(&pos);
        }
        let mut acc = self.constant_int(1)?;
        for _ in 0..e.min(64) {
            acc = self.mul_i(&acc, base)?;
        }
        Ok(acc)
    }

    fn abs_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let neg = self.gate_cell(-a.value, |gate, ctx| gate.neg(ctx, a.assigned));
        let zero = self.constant_int(0)?;
        let is_neg = self.lt_i(a, &zero)?;
        self.select_bit(&is_neg, &neg, a)
    }

    fn sign_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let zero = self.constant_int(0)?;
        let is_pos = self.lt_i(&zero, a)?;
        let is_neg = self.lt_i(a, &zero)?;
        self.sub_i(&is_pos, &is_neg)
    }

    fn inv_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let value = a.value.invert().unwrap_or(Fp::zero());
        Ok(self.gate_cell(value, |gate, ctx| gate.div_unsafe(ctx, QuantumCell::Constant(Fp::one()), a.assigned)))
    }

    fn bit_and_i(&mut self, _: &Self::CellRef, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("bit_and_i") }
    fn bit_or_i(&mut self, _: &Self::CellRef, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("bit_or_i") }
    fn bit_xor_i(&mut self, _: &Self::CellRef, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("bit_xor_i") }
    fn shl_i(&mut self, _: &Self::CellRef, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("shl_i") }
    fn shr_i(&mut self, _: &Self::CellRef, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("shr_i") }
    fn bit_not_i(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("bit_not_i") }

    fn add_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.add_i(a, b) }
    fn sub_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.sub_i(a, b) }

    fn mul_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let (result, rem) = kernel::fp_mul_rescale(a.value, b.value, precision_bits);
        let raw = self.mul_i(a, b)?;
        let result_cell = self.cell(result);
        let rem_cell = self.cell(rem);
        let scale = self.constant_int(crate::prove::field::quantization_scale(precision_bits) as i64)?;
        let recomposed = self.gate_cell(result * scale.value + rem, |gate, ctx| {
            gate.mul_add(ctx, result_cell.assigned, scale.assigned, rem_cell.assigned)
        });
        self.constrain_equal(&recomposed, &raw);
        let range = self.builder.range_chip();
        range.range_check(self.ctx(), rem_cell.assigned, precision_bits as usize);
        Ok(result_cell)
    }

    fn product_sum_f(&mut self, pairs: &[(Self::CellRef, Self::CellRef)]) -> Result<Self::CellRef, ProvingError> {
        let mut raw = self.constant_int(0)?;
        for (a, b) in pairs {
            raw = self.gate_cell(raw.value + a.value * b.value, |gate, ctx| {
                gate.mul_add(ctx, a.assigned, b.assigned, raw.assigned)
            });
        }
        let values: Vec<(Fp, Fp)> = pairs.iter().map(|(a, b)| (a.value, b.value)).collect();
        let (result, rem) = kernel::fp_product_sum_rescale(&values, self.params.precision_bits);
        let result_cell = self.cell(result);
        let rem_cell = self.cell(rem);
        let scale = self.constant_int(crate::prove::field::quantization_scale(self.params.precision_bits) as i64)?;
        let recomposed = self.gate_cell(result * scale.value + rem, |gate, ctx| {
            gate.mul_add(ctx, result_cell.assigned, scale.assigned, rem_cell.assigned)
        });
        self.constrain_equal(&recomposed, &raw);
        let precision_bits = self.params.precision_bits;
        let range = self.builder.range_chip();
        range.range_check(self.ctx(), rem_cell.assigned, precision_bits as usize);
        Ok(result_cell)
    }

    fn div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let scale = self.constant_int(crate::prove::field::quantization_scale(precision_bits) as i64)?;
        let scaled = self.mul_i(a, &scale)?;
        let result = kernel::fp_div_prescale(a.value, b.value, precision_bits);
        let result_cell = self.cell(result);
        let rem = scaled.value - b.value * result;
        let rem_cell = self.cell(rem);
        let recomposed = self.gate_cell(b.value * result + rem, |gate, ctx| {
            gate.mul_add(ctx, b.assigned, result_cell.assigned, rem_cell.assigned)
        });
        self.constrain_equal(&recomposed, &scaled);
        Ok(result_cell)
    }

    fn floor_div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.div_f(a, b) }
    fn mod_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.mod_i(a, b) }
    fn pow_f(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.pow_f_int(base, exp)
    }
    fn pow_f_int(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let max_exp = 8usize;
        let range = self.builder.range_chip();
        range.check_less_than_safe(self.ctx(), exp.assigned, (max_exp + 1) as u64);

        let mut powers = Vec::with_capacity(max_exp + 1);
        let mut acc = self.constant_float(1.0)?;
        powers.push(acc);
        for _ in 1..=max_exp {
            acc = self.mul_f(&acc, base)?;
            powers.push(acc);
        }

        let mut out = self.constant_float(0.0)?;
        for (i, power) in powers.iter().enumerate() {
            let idx = self.constant_int(i as i64)?;
            let hit = self.eq_i(exp, &idx)?;
            out = self.select_bit(&hit, power, &out)?;
        }
        Ok(out)
    }
    fn abs_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.abs_i(a) }
    fn sign_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.sign_i(a) }

    fn eq_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let value = if a.value == b.value { Fp::one() } else { Fp::zero() };
        let out = self.gate_cell(value, |gate, ctx| gate.is_equal(ctx, a.assigned, b.assigned));
        self.mark_bit(&out);
        Ok(out)
    }
    fn ne_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let eq = self.eq_i(a, b)?;
        self.logical_not(&eq)
    }
    fn lt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.signed_lt(a, b)
    }
    fn lt_i_bounded(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let value = if kernel::fp_to_i64(a.value) < kernel::fp_to_i64(b.value) { Fp::one() } else { Fp::zero() };
        let a_known = self.is_unsigned64_known(a);
        let b_known = self.is_unsigned64_known(b);
        let range = self.builder.range_chip();
        if !a_known {
            range.range_check(self.ctx(), a.assigned, 64);
            self.unsigned64_cells.insert(a.id);
        }
        if !b_known {
            range.range_check(self.ctx(), b.assigned, 64);
            self.unsigned64_cells.insert(b.id);
        }
        let assigned = range.is_less_than(self.ctx(), a.assigned, b.assigned, 64);
        let out = self.wrap(assigned, value);
        self.mark_bit(&out);
        Ok(out)
    }
    fn gt_i_bounded(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i_bounded(b, a) }
    fn lte_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let gt = self.gt_i(a, b)?;
        self.logical_not(&gt)
    }
    fn gt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i(b, a) }
    fn gte_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let lt = self.lt_i(a, b)?;
        self.logical_not(&lt)
    }
    fn eq_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.eq_i(a, b) }
    fn ne_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.ne_i(a, b) }
    fn lt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i(a, b) }
    fn lte_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let gt = self.gt_f(a, b)?;
        self.logical_not(&gt)
    }
    fn gt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i(b, a) }
    fn gte_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let lt = self.lt_f(a, b)?;
        self.logical_not(&lt)
    }

    fn sin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let a_abs = self.abs_i(a)?;
        let zero = self.constant_int(0)?;
        let a_sign = self.lt_i(a, &zero)?;
        let pi = self.constant_float(std::f64::consts::PI)?;
        let q_cell = self.floor_div_i(&a_abs, &pi)?;
        let r_cell = self.mod_i(&a_abs, &pi)?;
        let two = self.constant_int(2)?;
        let q_parity = self.mod_i(&q_cell, &two)?;

        let coefs: Vec<Fp> = kernel::SIN_COEFS.iter().map(|c| self.quantize_fp(*c)).collect();
        let sin_val = self.poly_eval_horner(&r_cell, &coefs)?;

        let both = self.mul_i(&a_sign, &q_parity)?;
        let two_both = self.add_i(&both, &both)?;
        let sum = self.add_i(&a_sign, &q_parity)?;
        let final_sign = self.sub_i(&sum, &two_both)?;
        self.assert_bit(&final_sign)?;
        self.signed_negate_if(&sin_val, &final_sign)
    }
    fn sinh_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("sinh_f") }
    fn cos_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let half_pi = self.constant_float(std::f64::consts::FRAC_PI_2)?;
        let shifted = self.add_f(a, &half_pi)?;
        self.sin_f(&shifted)
    }
    fn cosh_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("cosh_f") }
    fn tan_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("tan_f") }
    fn tanh_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("tanh_f") }
    fn sqrt_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let a_f64 = kernel::dequantize_from_fp(a.value, self.params.precision_bits);
        let y_f64 = if a_f64.is_finite() && a_f64 >= 0.0 {
            a_f64.sqrt()
        } else {
            0.0
        };
        let y = self.cell(self.quantize_fp(y_f64));
        let y_sq = self.mul_f(&y, &y)?;
        self.assert_close_f(&y_sq, a, 0.001)?;
        Ok(y)
    }
    fn exp_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("exp_f") }
    fn log_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("log_f") }
    fn arcsin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let a_f64 = kernel::dequantize_from_fp(a.value, self.params.precision_bits);
        let y_f64 = if (-1.0..=1.0).contains(&a_f64) {
            a_f64.asin()
        } else {
            0.0
        };
        let y = self.cell(self.quantize_fp(y_f64));
        let sin_y = self.sin_f(&y)?;
        self.assert_close_f(&sin_y, a, 0.001)?;
        Ok(y)
    }
    fn arccos_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("arccos_f") }
    fn arctan2_f(&mut self, _: &Self::CellRef, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("arctan2_f") }

    fn logical_and(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let ta = self.truthy_bit(a);
        let tb = self.truthy_bit(b);
        let out = self.gate_cell(ta.value * tb.value, |gate, ctx| gate.and(ctx, ta.assigned, tb.assigned));
        self.mark_bit(&out);
        Ok(out)
    }
    fn logical_or(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let ta = self.truthy_bit(a);
        let tb = self.truthy_bit(b);
        let out = self.gate_cell(ta.value + tb.value - ta.value * tb.value, |gate, ctx| gate.or(ctx, ta.assigned, tb.assigned));
        self.mark_bit(&out);
        Ok(out)
    }
    fn logical_not(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let ta = self.truthy_bit(a);
        let out = self.gate_cell(Fp::one() - ta.value, |gate, ctx| gate.not(ctx, ta.assigned));
        self.mark_bit(&out);
        Ok(out)
    }

    fn select(&mut self, cond: &Self::CellRef, if_true: &Self::CellRef, if_false: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let cond = self.truthy_bit(cond);
        self.select_bit(&cond, if_true, if_false)
    }
    fn select_bit(&mut self, cond: &Self::CellRef, if_true: &Self::CellRef, if_false: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.assert_bit(cond)?;
        let value = cond.value * if_true.value + (Fp::one() - cond.value) * if_false.value;
        // Zero-branch fusion: select(c, 0, x) == (1-c)*x and
        // select(c, x, 0) == c*x — one gate instead of two.
        let t_zero = self.is_const_cell(if_true) && if_true.value == Fp::zero();
        let f_zero = self.is_const_cell(if_false) && if_false.value == Fp::zero();
        let out = if t_zero {
            self.gate_cell(value, |gate, ctx| gate.mul_not(ctx, cond.assigned, if_false.assigned))
        } else if f_zero {
            self.gate_cell(value, |gate, ctx| gate.mul(ctx, cond.assigned, if_true.assigned))
        } else {
            self.gate_cell(value, |gate, ctx| gate.select(ctx, if_true.assigned, if_false.assigned, cond.assigned))
        };
        // The select gate pins out to one of the branches, so window facts
        // shared by both branches hold for the output as well.
        if self.is_signed64_known(if_true) && self.is_signed64_known(if_false) {
            self.signed64_cells.insert(out.id);
        }
        if self.is_unsigned64_known(if_true) && self.is_unsigned64_known(if_false) {
            self.unsigned64_cells.insert(out.id);
        }
        if let (Some((tl, th)), Some((fl, fh))) =
            (self.interval_of(if_true), self.interval_of(if_false))
        {
            self.value_interval.insert(out.id, (tl.min(fl), th.max(fh)));
        }
        Ok(out)
    }
    fn select_many_lane(&mut self, cond: &Self::CellRef, true_values: &[Self::CellRef], false_values: &[Self::CellRef], lane: usize) -> Result<Self::CellRef, ProvingError> {
        let cond = self.truthy_bit(cond);
        let mut key = Vec::with_capacity(1 + true_values.len() + false_values.len());
        key.push(cond.id);
        for cell in true_values.iter().chain(false_values.iter()) {
            key.push(cell.id);
        }
        if !self.select_many_cache.contains_key(&key) {
            let mut outs = Vec::with_capacity(true_values.len());
            for (t, f) in true_values.iter().zip(false_values) {
                outs.push(self.select_bit(&cond, t, f)?);
            }
            self.select_many_cache.insert(key.clone(), outs);
        }
        self.select_many_cache.get(&key).and_then(|v| v.get(lane)).copied().ok_or_else(|| ProvingError::synthesis("select_many lane out of bounds"))
    }

    fn int_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let scale = crate::prove::field::quantization_scale(self.params.precision_bits) as i64;
        let q = kernel::fp_to_i64(a.value).div_euclid(scale);
        Ok(self.cell(kernel::i64_to_fp(q)))
    }
    fn float_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let scale = self.constant_int(crate::prove::field::quantization_scale(self.params.precision_bits) as i64)?;
        self.mul_i(a, &scale)
    }
    fn bool_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { Ok(self.truthy_bit(a)) }

    fn read_input(&mut self, path: &InputPath, is_public: bool) -> Result<Self::CellRef, ProvingError> {
        let val = self.get_witness_fp_path(path).unwrap_or(Fp::zero());
        let cell = self.cell(val);
        if is_public {
            self.expose_public(&cell, &path.display())?;
        }
        Ok(cell)
    }
    fn read_external_result(&mut self, store_idx: u32, output_idx: u32) -> Result<Self::CellRef, ProvingError> {
        Ok(self.cell(self.get_external_fp(store_idx, output_idx).unwrap_or(Fp::zero())))
    }
    fn expose_public(&mut self, a: &Self::CellRef, _label: &str) -> Result<(), ProvingError> {
        if self.builder.assigned_instances.is_empty() {
            self.builder.set_instance_columns(1);
        }
        self.builder.assigned_instances[0].push(a.assigned);
        self.public_values.push(a.value);
        Ok(())
    }
    fn assert_true(&mut self, a: &Self::CellRef) -> Result<(), ProvingError> {
        self.assert_const(a, Fp::one());
        Ok(())
    }
    fn assert_u128_lt(&mut self, idx: &Self::CellRef, hi: &Self::CellRef) -> Result<(), ProvingError> {
        let lt = self.lt_i_bounded(idx, hi)?;
        self.assert_true(&lt)
    }
    fn assert_u128_lt_bit(&mut self, idx: &Self::CellRef, hi: u64, bit: u32, bits: u32) -> Result<Self::CellRef, ProvingError> {
        let key = (idx.id, hi, bits);
        if !self.u128_lt_bit_cache.contains_key(&key) {
            let range = self.builder.range_chip();
            range.check_less_than_safe(self.ctx(), idx.assigned, hi);
            let decomposed = GateChip::<Fp>::default().num_to_bits(self.ctx(), idx.assigned, bits as usize);
            let cells = decomposed
                .into_iter()
                .map(|assigned| {
                    let value = *assigned.value();
                    let cell = self.wrap(assigned, value);
                    self.mark_bit(&cell);
                    cell
                })
                .collect();
            self.u128_lt_bit_cache.insert(key, cells);
        }
        self.u128_lt_bit_cache.get(&key).and_then(|v| v.get(bit as usize)).copied().ok_or_else(|| ProvingError::synthesis("range bit index out of bounds"))
    }
    fn assert_non_neg_i(&mut self, a: &Self::CellRef) -> Result<(), ProvingError> {
        // For signed-i64-domain values, x >= 0 is exactly x in [0, 2^63):
        // negatives encode as field elements near p and fail the check.
        let range = self.builder.range_chip();
        range.range_check(self.ctx(), a.assigned, 63);
        // The check enforces a in [0, 2^63); tighten any known interval.
        let tightened = match self.interval_of(a) {
            Some((lo, hi)) => (lo.max(0), hi),
            None => (0, i64::MAX),
        };
        self.value_interval.insert(a.id, tightened);
        Ok(())
    }

    fn assert_bit(&mut self, a: &Self::CellRef) -> Result<(), ProvingError> {
        // Cells in bit_cell_ids are already constrained boolean (is_equal /
        // is_zero / comparison outputs, or a previous assert_bit) — only
        // structurally marked bits qualify, never the value==0/1 heuristic,
        // so the skip is witness-independent and sound.
        if self.bit_cell_ids.contains(&a.id) {
            return Ok(());
        }
        GateChip::<Fp>::default().assert_bit(self.ctx(), a.assigned);
        self.mark_bit(a);
        Ok(())
    }

    fn allocate_memory(&mut self, segment_id: u32, size: u32, init: i64) -> Result<(), ProvingError> {
        let init_fp = kernel::i64_to_fp(init);
        self.memories.insert(segment_id, vec![init_fp; size as usize]);
        self.memory_cells.insert(segment_id, vec![None; size as usize]);
        self.memory_init.insert(segment_id, init_fp);
        *self.write_version.entry(segment_id).or_insert(0) += 1;
        Ok(())
    }
    fn write_memory(&mut self, segment_id: u32, addr: &Self::CellRef, value: &Self::CellRef) -> Result<(), ProvingError> {
        *self.write_version.entry(segment_id).or_insert(0) += 1;
        if self.is_const_cell(addr) {
            let a = kernel::fp_to_i64(addr.value) as usize;
            let mem = self.memories.get_mut(&segment_id).ok_or_else(|| ProvingError::synthesis("segment not allocated"))?;
            if a >= mem.len() {
                return Err(ProvingError::synthesis("memory write out of bounds"));
            }
            mem[a] = value.value;
            if let Some(cells) = self.memory_cells.get_mut(&segment_id) {
                cells[a] = Some(*value);
            }
            return Ok(());
        }

        let len = self
            .memories
            .get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("segment not allocated"))?
            .len();
        let range = self.builder.range_chip();
        range.check_less_than_safe(self.ctx(), addr.assigned, len as u64);

        self.dynamic_writes
            .entry(segment_id)
            .or_default()
            .push((*addr, *value));
        let a = kernel::fp_to_i64(addr.value) as usize;
        if let Some(mem) = self.memories.get_mut(&segment_id) {
            if a < mem.len() {
                mem[a] = value.value;
            }
        }
        Ok(())
    }
    fn read_memory(&mut self, segment_id: u32, addr: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let dynamic_writes = self
            .dynamic_writes
            .get(&segment_id)
            .cloned()
            .unwrap_or_default();

        if self.is_const_cell(addr) {
            let a = kernel::fp_to_i64(addr.value) as usize;
            let init = self.memory_init.get(&segment_id).copied().unwrap_or(Fp::zero());
            let base = self
                .memory_cells
                .get(&segment_id)
                .and_then(|cells| cells.get(a))
                .copied()
                .flatten()
                // Never-const-written slot: its pre-overlay content is the
                // init value by construction; a constant cell keeps it
                // constrained (a witness cell here would be a free variable).
                .unwrap_or_else(|| self.const_cell(init));
            let mut out = base;
            for (write_addr, write_value) in dynamic_writes {
                let hit = self.overlay_hit(&write_addr, addr)?;
                out = self.select_bit(&hit, &write_value, &out)?;
            }
            return Ok(out);
        }

        let len = self
            .memories
            .get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("segment not allocated"))?
            .len();
        let init = self.memory_init.get(&segment_id).copied().unwrap_or(Fp::zero());

        // Affine view of the address (identity view when no lineage is
        // known). Two lowering strategies:
        //
        // Two dynamic roots (windowed gathers, addr = r1*s1 + r2*s2 + c):
        // factored two-stage gather. Stage-1 planes Σ_v eq(r1,v)*mem[v*s1+t]
        // are cached and shared across reads; each read then muxes the
        // s1/s2 digit choices of r2. Both roots are pinned one-hot and the
        // full address is range-checked (an out-of-bounds (v,w) pair would
        // otherwise silently select the init value).
        //
        // One root (strided gathers, addr = root*s + c): mux only the slot
        // indices reachable from the root; eq(root, v) indicators are shared
        // across reads. The one-hot membership over the in-bounds choice set
        // subsumes the range check.
        let lin = self.affine_lin_of(addr);
        let mut out: Option<Halo2BaseCell> = None;

        if lin.terms.len() == 2 {
            let (mut t1, mut t2) = (lin.terms[0], lin.terms[1]);
            if t1.1.abs() < t2.1.abs() {
                std::mem::swap(&mut t1, &mut t2);
            }
            let ((root1, s1), (root2, s2)) = (t1, t2);
            if s1 > 0 && s2 > 0 && s1 % s2 == 0 && s1 / s2 >= 2 {
                // Digit-canonical choices for the minor root: residuals that
                // stay inside one major stride. Matches Zinnia's clamped
                // in-bounds index lowering; a kernel violating it fails the
                // one-hot assert at verify time rather than proving silently.
                let w_choices: Vec<i64> = (0..s1 / s2)
                    .filter(|w| {
                        let t = w * s2 + lin.offset;
                        (0..len as i64).any(|idx| (idx - t) % s1 == 0)
                    })
                    .collect();
                let v_union: Vec<i64> = {
                    let mut vs: Vec<i64> = Vec::new();
                    for w in &w_choices {
                        let t = w * s2 + lin.offset;
                        for idx in 0..len as i64 {
                            let num = idx - t;
                            if num % s1 == 0 && !vs.contains(&(num / s1)) {
                                vs.push(num / s1);
                            }
                        }
                    }
                    vs.sort_unstable();
                    vs
                };
                if !w_choices.is_empty() && !v_union.is_empty() {
                    let range = self.builder.range_chip();
                    range.check_less_than_safe(self.ctx(), addr.assigned, len as u64);
                    let mut acc = self.cell(init);
                    for w in &w_choices {
                        let t = w * s2 + lin.offset;
                        let plane = self.gathered_plane(segment_id, &root1, s1, t, len, init)?;
                        let ind = self.root_indicator(&root2, *w)?;
                        acc = self.select_bit(&ind, &plane, &acc)?;
                    }
                    self.assert_one_hot(&root2, &w_choices)?;
                    self.assert_one_hot(&root1, &v_union)?;
                    out = Some(acc);
                }
            }
        }

        if out.is_none() {
            let (root, scale, offset) = match lin.terms.len() {
                1 => (lin.terms[0].0, lin.terms[0].1, lin.offset),
                _ => (*addr, 1, 0),
            };
            // Constraint-derived root interval shrinks the choice set: a
            // unit-stride gather (addr = base + col, col in [0,W)) muxes W
            // slots instead of the whole segment. The membership assert
            // below stays sound — the interval is enforced by the root's
            // defining gates, so the true value is always inside it.
            let root_iv = self.interval_of(&root);
            let mut choices: Vec<(i64, usize)> = Vec::new(); // (root value v, slot idx)
            for idx in 0..len as i64 {
                let num = idx - offset;
                if scale != 0 && num % scale == 0 {
                    let v = num / scale;
                    if let Some((lo, hi)) = root_iv {
                        if v < lo || v > hi {
                            continue;
                        }
                    }
                    choices.push((v, idx as usize));
                }
            }
            if choices.is_empty() {
                return Err(ProvingError::synthesis(
                    "dynamic memory read can never be in bounds",
                ));
            }
            let mut acc = self.cell(init);
            for (v, idx) in &choices {
                let ind = self.root_indicator(&root, *v)?;
                let slot = self
                    .memory_cells
                    .get(&segment_id)
                    .and_then(|cells| cells.get(*idx))
                    .copied()
                    .flatten()
                    .unwrap_or_else(|| self.const_cell(init));
                acc = self.select_bit(&ind, &slot, &acc)?;
            }
            // Soundness: the gates that built `addr` already enforce
            // addr == root * scale + offset; pinning root one-hot onto the
            // in-bounds choices implies addr in bounds, replacing the
            // per-read range check.
            let vs: Vec<i64> = choices.iter().map(|(v, _)| *v).collect();
            self.assert_one_hot(&root, &vs)?;
            out = Some(acc);
        }

        let mut out = out.expect("dynamic read lowering produced a value");
        for (write_addr, write_value) in dynamic_writes {
            let hit = self.overlay_hit(&write_addr, addr)?;
            out = self.select_bit(&hit, &write_value, &out)?;
        }
        Ok(out)
    }
    fn memory_trace_emit(&mut self, _: u32, _: bool, _: &[Self::CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError> { Ok(()) }

    fn allocate_dynamic_ndarray_meta(&mut self, array_id: u32, _: &str, max_length: u32, max_rank: u32, max_shape: &[u32]) -> Result<(), ProvingError> {
        let axis_bounds = if max_shape.is_empty() { vec![max_length; max_rank as usize] } else { max_shape.to_vec() };
        self.dyn_meta_bounds.insert(array_id, (max_length, axis_bounds));
        self.allocate_memory(10000 + array_id, max_length, 0)
    }
    fn witness_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: &str, _: &[Self::CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn assert_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: u32, _: &[Self::CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn dynamic_ndarray_get_item(&mut self, array_id: u32, _: u32, index: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.read_memory(10000 + array_id, index)
    }
    fn dynamic_ndarray_set_item(&mut self, array_id: u32, _: u32, index: &Self::CellRef, value: &Self::CellRef) -> Result<(), ProvingError> {
        self.write_memory(10000 + array_id, index, value)
    }

    fn poseidon_hash(&mut self, inputs: &[Self::CellRef]) -> Result<Self::CellRef, ProvingError> {
        let mut state = [self.constant_int(0)?, self.constant_int(0)?, self.constant_int(0)?];
        for (i, input) in inputs.iter().enumerate() {
            let idx = i % 2;
            state[idx] = self.add_i(&state[idx], input)?;
            if idx == 1 || i == inputs.len() - 1 {
                for _ in 0..4 {
                    for slot in &mut state {
                        let x2 = self.mul_i(slot, slot)?;
                        let x4 = self.mul_i(&x2, &x2)?;
                        *slot = self.mul_i(&x4, slot)?;
                    }
                    state = [
                        self.add_i(&state[0], &state[1])?,
                        self.add_i(&state[1], &state[2])?,
                        self.add_i(&state[2], &state[0])?,
                    ];
                }
            }
        }
        Ok(state[0])
    }
    fn eq_hash(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.eq_i(a, b) }
}
