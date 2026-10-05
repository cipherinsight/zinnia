use std::collections::{HashMap, HashSet};

use halo2_base::{
    gates::{
        circuit::builder::BaseCircuitBuilder,
        flex_gate::{GateChip, GateInstructions},
        range::{RangeChip, RangeInstructions},
    },
    halo2_proofs::halo2curves::bn256::Fr as Fp,
    AssignedValue, QuantumCell,
};

use crate::circuit_input::{InputPath, ResolvedWitness};
use crate::prove::error::ProvingError;
use crate::prove::kernel::{self, Rounding};
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
    segments: HashMap<u32, Segment>,
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
    // (segment, slots version, root id, root scale, residual offset).
    plane_cache: HashMap<(u32, u64, u64, i64, i64), Halo2BaseCell>,
    // Cells already constrained to the signed-64 window (x + 2^63 < 2^64)
    // resp. the unsigned-64 window (x < 2^64). Comparisons skip their input
    // range checks for cells whose window is already enforced; selects
    // propagate the facts (the output provably equals one branch).
    signed64_cells: HashSet<u64>,
    unsigned64_cells: HashSet<u64>,
    // (sign bit, magnitude) per cell id, see `sign_and_abs`.
    sign_abs_cache: HashMap<u64, (Halo2BaseCell, Halo2BaseCell)>,
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
    /// First assert/bit constraint whose concrete witness violates its
    /// gate, captured during synthesis when a witness is present.
    /// halo2's `create_proof` happily emits a (verify-rejected) proof for
    /// an unsatisfiable witness; the prove path queries this to surface
    /// `assert(False)`-style failures before spending MSM time. Populated
    /// only when `witness.is_some()` so the witness-less verify-side
    /// re-keygen (placeholder zeros) never trips it.
    first_unsat: Option<String>,
    /// One `GateChip`, constructed once in `new()` and reused for every IR
    /// op. Constructing a `GateChip` precomputes a 254-entry `pow_of_two`
    /// table and a 1024-entry `field_element_cache` (~1.3k Montgomery
    /// conversions); doing that per op dominated gate emission. The chip is
    /// stateless after construction, so a single shared instance is sound.
    gate: GateChip<Fp>,
    /// Lazily-built `RangeChip`, sharing the builder's `LookupAnyManager`s
    /// (cheap `Arc` clones). Built on first range use so circuits without
    /// range checks pay no setup. Byte-identical to `builder.range_chip()`.
    range: Option<RangeChip<Fp>>,
    /// Test-only hook that perturbs prover hints (see `hint`).
    #[cfg(test)]
    tamper: tamper::HintTamper,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum OverlayReadKey {
    Cell(u64),
    Const(i64),
}

/// One allocated memory segment.
///
/// Program-order rule (M3): writes at constant addresses made before the
/// segment's first dynamic-address write go straight into `slots` (pure
/// wiring, no constraints). From the first dynamic-address write on, every
/// write, constant or dynamic, is appended to `overlay`, and every read
/// applies the overlay entries in that order on top of the slot value. The
/// last write whose address equals the read address therefore wins, which
/// is the program-order semantics of the mock backend.
struct Segment {
    init: Fp,
    /// Constant-address writes before the first dynamic write; `None` means
    /// never written (the slot holds `init`).
    slots: Vec<Option<Halo2BaseCell>>,
    /// `(address, value)` of every write since the first dynamic write.
    overlay: Vec<(Halo2BaseCell, Halo2BaseCell)>,
    /// Bumped whenever `slots` changes; keys the gathered-plane cache.
    slots_version: u64,
}

/// Whether an overlay write affects a read: decided at synthesis time when
/// both addresses are constants, otherwise by an equality bit.
enum OverlayHit {
    Always,
    Never,
    When(Halo2BaseCell),
}

/// The divisor of a constrained division, classified by what the circuit
/// already knows about it (see `div_rem`).
#[derive(Clone, Copy)]
enum Divisor {
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
            segments: HashMap::new(),
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
            signed64_cells: HashSet::new(),
            unsigned64_cells: HashSet::new(),
            sign_abs_cache: HashMap::new(),
            overlay_eq_cache: HashMap::new(),
            value_interval: HashMap::new(),
            prov: provenance,
            vid_of_cell: HashMap::new(),
            cell_of_vid: HashMap::new(),
            first_unsat: None,
            gate: GateChip::<Fp>::new(),
            range: None,
            #[cfg(test)]
            tamper: tamper::HintTamper::default(),
        }
    }

    /// The shared `RangeChip`, built on first use. Identical to
    /// `self.builder.range_chip()` (same `lookup_bits`, same `Arc`-shared
    /// `LookupAnyManager`s), but constructed at most once. The chip is
    /// stateless w.r.t. emission (lookups are queued through the shared
    /// managers), so a returned clone behaves identically to the cached
    /// instance — clone out to dodge borrow conflicts with `self.builder`.
    fn range_chip(&mut self) -> RangeChip<Fp> {
        if self.range.is_none() {
            // `builder.range_chip()` = `RangeChip::new(lookup_bits,
            // lookup_manager.clone())`; build it once, then hand out cheap
            // clones (Arc-shared managers, memcpy'd limb bases).
            self.range = Some(self.builder.range_chip());
        }
        self.range.as_ref().unwrap().clone()
    }

    pub fn public_values(&self) -> &[Fp] {
        &self.public_values
    }

    /// The first constraint whose witness value violated its gate during
    /// synthesis, or `None` if every recorded assert/bit constraint is
    /// satisfied (or no witness was bound). See `first_unsat`.
    pub fn first_unsat(&self) -> Option<&str> {
        self.first_unsat.as_deref()
    }

    /// Record (once) that a recorded constraint is unsatisfied by the
    /// concrete witness. No-op when no witness is bound.
    fn note_unsat(&mut self, detail: impl FnOnce() -> String) {
        if self.witness.is_some() && self.first_unsat.is_none() {
            self.first_unsat = Some(detail());
        }
    }

    fn ctx(&mut self) -> &mut halo2_base::Context<Fp> {
        self.builder.main(0)
    }

    /// A free witness cell. Used for circuit inputs; prover-computed
    /// intermediate values go through [`Self::hint`] instead.
    fn cell(&mut self, value: Fp) -> Halo2BaseCell {
        let assigned = self.ctx().load_witness(value);
        self.wrap(assigned, value)
    }

    /// A value the prover computes outside the circuit (a quotient, a
    /// square root, a sign bit). The cell is free when loaded; every caller
    /// must add constraints that leave exactly one admissible value for it
    /// given the cells it was computed from, and say why next to them. The
    /// tamper tests (`synthesizer_tests.rs`) replace hints with other values
    /// and check that the circuit then rejects the witness.
    fn hint(&mut self, value: Fp) -> Halo2BaseCell {
        #[cfg(test)]
        let value = self.tamper.apply(value);
        self.cell(value)
    }

    /// A boolean hint, constrained to {0, 1} and marked as a bit.
    fn hint_bit(&mut self, value: bool) -> Halo2BaseCell {
        let bit = self.hint(if value { Fp::one() } else { Fp::zero() });
        self.gate.assert_bit(self.builder.main(0), bit.assigned);
        self.mark_bit(&bit);
        bit
    }

    /// Constrain `0 <= x < 2^bits`.
    fn range_check(&mut self, x: &Halo2BaseCell, bits: usize) {
        let range = self.range_chip();
        range.range_check(self.ctx(), x.assigned, bits);
    }

    /// `bits` rounded up to a multiple of the lookup width: a range check
    /// of that many bits costs no more limbs and needs no extra cell for a
    /// partial top limb.
    fn lookup_aligned(&mut self, bits: usize) -> usize {
        let lookup = self.range_chip().lookup_bits();
        bits.div_ceil(lookup) * lookup
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
        // Only structural sources of bit-ness qualify — the skip decision
        // must be identical for keygen (Some/None witness) and prove, or the
        // gate count diverges and the VK no longer matches the proof. The
        // value heuristic (cell.value == 0/1) is witness-dependent and was
        // deliberately removed (code review C03). A constant cell whose fixed
        // value is exactly 0 or 1 is witness-independent and so still counts.
        if self.bit_cell_ids.contains(&cell.id) {
            return true;
        }
        if self.is_const_cell(cell) && (cell.value == Fp::zero() || cell.value == Fp::one()) {
            return true;
        }
        cell.assigned
            .cell
            .map(|ptr| self.bit_cells.contains(&ptr))
            .unwrap_or(false)
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
        let version = self.segment(segment_id)?.slots_version;
        let key = (segment_id, version, root.id, scale, t);
        if let Some(plane) = self.plane_cache.get(&key) {
            return Ok(*plane);
        }
        let mut out = self.const_cell(init);
        for idx in 0..len as i64 {
            let num = idx - t;
            if num % scale != 0 {
                continue;
            }
            let ind = self.root_indicator(root, num / scale)?;
            let slot = self.slot_cell(segment_id, idx as usize)?;
            out = self.select_bit(&ind, &slot, &out)?;
        }
        self.plane_cache.insert(key, out);
        Ok(out)
    }

    /// Whether an overlay write at `write_addr` is seen by a read at
    /// `read_addr`. Two constant addresses are compared at synthesis time;
    /// otherwise an eq bit is emitted, cached so repeated reads against the
    /// same pending write reuse the constraint.
    fn overlay_hit(
        &mut self,
        write_addr: &Halo2BaseCell,
        read_addr: &Halo2BaseCell,
    ) -> Result<OverlayHit, ProvingError> {
        if self.is_const_cell(write_addr) && self.is_const_cell(read_addr) {
            return Ok(if write_addr.value == read_addr.value {
                OverlayHit::Always
            } else {
                OverlayHit::Never
            });
        }
        let key = if self.is_const_cell(read_addr) {
            OverlayReadKey::Const(kernel::fp_to_i64(read_addr.value))
        } else {
            OverlayReadKey::Cell(read_addr.id)
        };
        if let Some(hit) = self.overlay_eq_cache.get(&(write_addr.id, key)) {
            return Ok(OverlayHit::When(*hit));
        }
        let hit = self.eq_i(&write_addr.clone(), &read_addr.clone())?;
        self.overlay_eq_cache.insert((write_addr.id, key), hit);
        Ok(OverlayHit::When(hit))
    }

    /// Apply the segment's overlay writes, in program order, to `base`
    /// (the value the read address holds before the first dynamic write).
    fn apply_overlay(
        &mut self,
        segment_id: u32,
        read_addr: &Halo2BaseCell,
        base: Halo2BaseCell,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let overlay = self.segment(segment_id)?.overlay.clone();
        let mut out = base;
        for (write_addr, write_value) in overlay {
            out = match self.overlay_hit(&write_addr, read_addr)? {
                OverlayHit::Always => write_value,
                OverlayHit::Never => out,
                OverlayHit::When(hit) => self.select_bit(&hit, &write_value, &out)?,
            };
        }
        Ok(out)
    }

    fn segment(&self, segment_id: u32) -> Result<&Segment, ProvingError> {
        self.segments
            .get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("segment not allocated"))
    }

    /// The cell a slot holds before the first dynamic write: its last
    /// constant-address write, or the segment's init value as a constant
    /// (a witness cell here would be a free variable). Addresses past the
    /// end read the init value, as in the mock backend.
    fn slot_cell(&mut self, segment_id: u32, idx: usize) -> Result<Halo2BaseCell, ProvingError> {
        let segment = self.segment(segment_id)?;
        let (written, init) = (segment.slots.get(idx).copied().flatten(), segment.init);
        Ok(written.unwrap_or_else(|| self.const_cell(init)))
    }

    fn gate_cell(
        &mut self,
        value: Fp,
        f: impl FnOnce(&GateChip<Fp>, &mut halo2_base::Context<Fp>) -> AssignedValue<Fp>,
    ) -> Halo2BaseCell {
        // Disjoint field borrows: `&self.gate` and `self.builder.main(0)`
        // touch different fields, so the borrow checker allows both live at
        // once (going through `self.ctx()` would borrow all of `self`).
        let assigned = f(&self.gate, self.builder.main(0));
        self.wrap(assigned, value)
    }

    fn truthy_bit(&mut self, a: &Halo2BaseCell) -> Halo2BaseCell {
        if self.is_bit(a) {
            return *a;
        }
        let iz = self.is_zero_bit(a);
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
        self.gate.assert_is_const(self.builder.main(0), &a.assigned, &c);
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

    /// `x + 2^63`, constrained to `[0, 2^64)` (once per cell), which puts
    /// `x` in the signed 64-bit window `[-2^63, 2^63)`.
    fn signed64_shifted(&mut self, x: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let known = self.is_signed64_known(x);
        let offset = self.const_cell(Fp::from(1u64 << 63));
        let shifted = self.add_i(x, &offset)?;
        if !known {
            self.range_check(&shifted, 64);
            self.signed64_cells.insert(x.id);
        }
        Ok(shifted)
    }

    /// Constrain `x` to the signed 64-bit window (no-op if already known).
    fn ensure_signed64(&mut self, x: &Halo2BaseCell) -> Result<(), ProvingError> {
        if !self.is_signed64_known(x) {
            self.signed64_shifted(x)?;
        }
        Ok(())
    }

    fn signed_lt(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let value = if kernel::fp_to_i64(a.value) < kernel::fp_to_i64(b.value) {
            Fp::one()
        } else {
            Fp::zero()
        };
        let a_shifted = self.signed64_shifted(a)?;
        let b_shifted = self.signed64_shifted(b)?;
        let range = self.range_chip();
        let assigned = range.is_less_than(self.ctx(), a_shifted.assigned, b_shifted.assigned, 64);
        let out = self.wrap(assigned, value);
        self.mark_bit(&out);
        Ok(out)
    }

    /// `[x == 0]` as a marked bit (halo2-base's inverse-witness gadget).
    fn is_zero_bit(&mut self, x: &Halo2BaseCell) -> Halo2BaseCell {
        let value = if x.value == Fp::zero() { Fp::one() } else { Fp::zero() };
        let assigned = self.gate.is_zero(self.builder.main(0), x.assigned);
        let bit = self.wrap(assigned, value);
        self.mark_bit(&bit);
        bit
    }

    /// `(s, s ? -x : x)` for a hinted sign bit `s = [x < 0]`. The sign is
    /// NOT pinned here: the caller must constrain the magnitude into a
    /// window far below p, which rules out the wrong sign (it would make
    /// the magnitude the field negation of a small number, about p).
    fn hinted_sign_and_magnitude(
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
    fn sign_and_abs(&mut self, x: &Halo2BaseCell) -> Result<(Halo2BaseCell, Halo2BaseCell), ProvingError> {
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
            (sign, magnitude)
        };
        self.sign_abs_cache.insert(x.id, pair);
        Ok(pair)
    }

    /// Constrain `0 <= x < n` for a constant `n >= 1`.
    ///
    /// Soundness: with `w >= bit_length(n - 1)`, the checks `x ∈ [0, 2^w)`
    /// and `n - 1 - x ∈ [0, 2^w)` both hold only if `x <= n - 1`: a negative
    /// `n - 1 - x` is a field element above `p - 2^w`. For a power of two
    /// `n = 2^k` the first check with `w = k` is already exact.
    fn assert_below_constant(&mut self, x: &Halo2BaseCell, n: u128) {
        debug_assert!(n >= 1);
        if n.is_power_of_two() {
            self.range_check(x, n.trailing_zeros() as usize);
            return;
        }
        let bits = self.lookup_aligned((128 - (n - 1).leading_zeros()) as usize);
        self.range_check(x, bits);
        let n_minus_one = kernel::i128_to_fp((n - 1) as i128);
        let gap = self.gate_cell(n_minus_one - x.value, |gate, ctx| {
            gate.sub(ctx, QuantumCell::Constant(n_minus_one), x.assigned)
        });
        self.range_check(&gap, bits);
    }

    /// Constrain `0 <= x < n` for a cell `n`. Both `x` and `n - 1 - x` are
    /// range-checked to 64 bits, so `n - 1`, their sum, lies in `[0, 2^65)`
    /// over the integers: the check also proves `1 <= n <= 2^65`.
    fn assert_below_cell(&mut self, x: &Halo2BaseCell, n: &Halo2BaseCell) {
        self.range_check(x, 64);
        let gap = self.gate_cell(n.value - Fp::one() - x.value, |gate, ctx| {
            let n_minus_x = gate.sub(ctx, n.assigned, x.assigned);
            gate.dec(ctx, n_minus_x)
        });
        self.range_check(&gap, 64);
    }

    /// The divisor kind of `b` for [`Self::div_rem_nonzero`]: a constant in
    /// the signed 64-bit window, or `None` for a cell to be handled
    /// dynamically.
    fn constant_divisor(&self, b: &Halo2BaseCell) -> Option<i128> {
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
    fn div_rem(
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
    ///     [`Self::assert_below_constant`]; for a cell it is
    ///     [`Self::assert_below_cell`] against `|b|`, which also proves
    ///     `1 <= |B| <= 2^65` and with it the hinted sign of `b`.
    ///
    /// Soundness: `|B·Q + R| < 2^65·2^63 + 2^65`, far below p/2, so
    /// `A = B·Q + R` holds over the integers. A second solution would give
    /// `B(Q - Q') = R' - R` with `|R' - R| < |B|` and equal remainder signs,
    /// so `Q = Q'` and `R = R'`: `q` and `r` are functions of `(a, b)`. An
    /// input whose true quotient leaves the window has no witness.
    fn div_rem_nonzero(
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
            (Divisor::Constant(c), _) => self.assert_below_constant(&r_adj, c.unsigned_abs()),
            (_, Some((_, b_magnitude))) => self.assert_below_cell(&r_adj, &b_magnitude),
            (Divisor::Positive(b), _) => self.assert_below_cell(&r_adj, &b),
            (Divisor::NonZero(_), None) => unreachable!("non-zero divisor without a sign"),
        }
        Ok((q, r))
    }

    /// `floor(raw / 2^P)` for the raw product of Q values (`mul_f`,
    /// `product_sum_f`); `q_value` is the quotient the kernel computed.
    ///
    /// Soundness: `rem = raw - q*2^P` (gate) is range-checked to
    /// `[0, 2^P)` and `q` to the signed 64-bit window, so `q*2^P + rem` lies
    /// in `[-2^95, 2^95)`: it equals `raw` over the integers and `q` is the
    /// unique floor. Without the window check every remainder in
    /// `[0, 2^P)` would be accepted, each with its own (huge) `q`.
    fn rescale(&mut self, raw: &Halo2BaseCell, q_value: Fp) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let scale = kernel::scale_fp(precision_bits);
        let q = self.hint(q_value);
        let rem = self.gate_cell(raw.value - q.value * scale, |gate, ctx| {
            gate.sub_mul(ctx, raw.assigned, q.assigned, QuantumCell::Constant(scale))
        });
        self.range_check(&rem, precision_bits as usize);
        self.ensure_signed64(&q)?;
        Ok(q)
    }

    /// `floor(sqrt(x * 2^P))`, the Q-encoded square root, for a cell `x`
    /// the caller has proven to lie in `[0, 2^64)`.
    ///
    /// Soundness: with `X = x * 2^P`, the hint `y` and `d = X - y^2` (gate),
    /// the checks `y ∈ [0, 2^wy)`, `d ∈ [0, 2^wd)` and `2y - d ∈ [0, 2^wd)`
    /// give `X = y^2 + d` over the integers (both terms are far below p) and
    /// `0 <= d <= 2y`, i.e. `y^2 <= X < (y + 1)^2`. Exactly one `y >= 0`
    /// satisfies that, so `y` is a function of `x`. `wy` and `wd` are sized
    /// for the largest `X < 2^(64 + P)`.
    fn sqrt_nonneg(&mut self, x: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let scale = kernel::scale_fp(precision_bits);
        let big_x = self.gate_cell(x.value * scale, |gate, ctx| {
            gate.mul(ctx, x.assigned, QuantumCell::Constant(scale))
        });
        let y = self.hint(kernel::fp_sqrt(x.value, precision_bits));
        let d = self.gate_cell(big_x.value - y.value * y.value, |gate, ctx| {
            gate.sub_mul(ctx, big_x.assigned, y.assigned, y.assigned)
        });
        let root_bits = 32 + precision_bits.div_ceil(2) as usize;
        let wy = self.lookup_aligned(root_bits + 1);
        let wd = self.lookup_aligned(root_bits + 2);
        self.range_check(&y, wy);
        self.range_check(&d, wd);
        let two_y = self.add_i(&y, &y)?;
        let slack = self.sub_i(&two_y, &d)?;
        self.range_check(&slack, wd);
        Ok(y)
    }

    /// Horner evaluation with leading-first Q coefficients, every product
    /// rounded down by `mul_f`: the algorithm of `kernel::horner_eval`.
    fn poly_eval_horner(&mut self, x: &Halo2BaseCell, coefs: &[Fp]) -> Result<Halo2BaseCell, ProvingError> {
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

    /// The slot value at a dynamic address, before overlay writes: a mux
    /// over the slots reachable from the address's affine view.
    ///
    /// Two dynamic roots (windowed gathers, addr = r1*s1 + r2*s2 + c):
    /// factored two-stage gather. Stage-1 planes Σ_v eq(r1,v)*mem[v*s1+t]
    /// are cached and shared across reads; each read then muxes the
    /// s1/s2 digit choices of r2. Both roots are pinned one-hot and the
    /// full address is range-checked (an out-of-bounds (v,w) pair would
    /// otherwise silently select the init value).
    ///
    /// One root (strided gathers, addr = root*s + c): mux only the slot
    /// indices reachable from the root; eq(root, v) indicators are shared
    /// across reads. The one-hot membership over the in-bounds choice set
    /// subsumes the range check.
    fn gather_slots(&mut self, segment_id: u32, addr: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let len = self.segment(segment_id)?.slots.len();
        let init = self.segment(segment_id)?.init;
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
                    let range = self.range_chip();
                    range.check_less_than_safe(self.ctx(), addr.assigned, len as u64);
                    let mut acc = self.const_cell(init);
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
            let mut acc = self.const_cell(init);
            for (v, idx) in &choices {
                let ind = self.root_indicator(&root, *v)?;
                let slot = self.slot_cell(segment_id, *idx)?;
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

        Ok(out.expect("dynamic read lowering produced a value"))
    }

    /// `[x < c]` as a marked bit, for an `x` the caller has proven to lie
    /// in `[0, 2^bits)` and a constant `c < 2^bits` (the precondition of
    /// halo2-base's `is_less_than`).
    fn below_constant_bit(&mut self, x: &Halo2BaseCell, c: i128, bits: usize) -> Halo2BaseCell {
        let c_fp = kernel::i128_to_fp(c);
        let range = self.range_chip();
        let assigned = range.is_less_than(self.ctx(), x.assigned, QuantumCell::Constant(c_fp), bits);
        let value = if kernel::fp_to_i128(x.value).is_some_and(|v| v < c) { Fp::one() } else { Fp::zero() };
        let bit = self.wrap(assigned, value);
        self.mark_bit(&bit);
        bit
    }

    fn negate(&mut self, value: &Halo2BaseCell) -> Halo2BaseCell {
        self.gate_cell(-value.value, |gate, ctx| gate.neg(ctx, value.assigned))
    }

    /// `sign ? -value : value` for a bit `sign`.
    fn negate_if(&mut self, value: &Halo2BaseCell, sign: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let neg = self.negate(value);
        self.select_bit(sign, &neg, value)
    }

    /// `base^e` for a constant `e >= 0`, by square-and-multiply.
    fn pow_const(&mut self, base: &Halo2BaseCell, e: u64) -> Result<Halo2BaseCell, ProvingError> {
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

    /// The IR's `DivI`: field division `a * b^-1`, total, with a zero
    /// divisor giving 0 as in the mock backend.
    ///
    /// Soundness: `b' = b + [b == 0]` is non-zero and `a' = a * [b != 0]`;
    /// the hint `q` is pinned by `q * b' == a'`, whose only solution is
    /// `a' / b'` (0 when `b = 0`). halo2-base's `div_unsafe` is not used: it
    /// panics on a zero divisor during keygen and leaves `0 / 0` free.
    fn div_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let b_is_zero = self.is_zero_bit(b);
        let b_nonzero = self.add_i(b, &b_is_zero)?;
        let a_masked = self.gate_cell((Fp::one() - b_is_zero.value) * a.value, |gate, ctx| {
            gate.mul_not(ctx, b_is_zero.assigned, a.assigned)
        });
        let q_value = a_masked.value * b_nonzero.value.invert().unwrap_or(Fp::zero());
        let q = self.hint(q_value);
        let product = self.mul_i(&q, &b_nonzero)?;
        self.constrain_equal(&product, &a_masked);
        Ok(q)
    }

    /// Python floor division `a // b` (remainder has the divisor's sign);
    /// see `div_rem` for the constraints and the zero-divisor convention.
    fn floor_div_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Ok(self.div_rem(a, b, Rounding::Floor)?.0)
    }

    /// `a // b` for operands the producer bounds by `2^divisor_bits` and
    /// keeps non-negative (`IR::FloorDivIPinned`, the `-1` dimension of a
    /// runtime reshape). The quotient, the remainder and `b - 1 - r` are
    /// range-checked to that width instead of 64 bits. The checks are what
    /// enforce the bound: if the producer were wrong the circuit would be
    /// unsatisfiable, not unsound. Soundness as in `div_rem_nonzero`:
    /// `r, b' - 1 - r ∈ [0, 2^w)` give `1 <= b' <= 2^(w+1)` and `r < b'`, and
    /// `b'·q + r < 2^(2w+1) + 2^w` stays far below p. A zero divisor maps to
    /// the division `0 / 1`, as in `div_rem`.
    fn floor_div_i_pinned(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
        divisor_bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        let bits = self.lookup_aligned(divisor_bits as usize);
        if bits >= 64 {
            return self.floor_div_i(a, b);
        }
        let b_is_zero = self.is_zero_bit(b);
        let b_nonzero = self.add_i(b, &b_is_zero)?;
        let a_masked = self.gate_cell((Fp::one() - b_is_zero.value) * a.value, |gate, ctx| {
            gate.mul_not(ctx, b_is_zero.assigned, a.assigned)
        });
        let q_value = match (kernel::fp_to_i128(a_masked.value), kernel::fp_to_i128(b_nonzero.value)) {
            (Some(av), Some(bv)) => kernel::div_rem_or_zero(av, bv, Rounding::Floor).0,
            _ => 0,
        };
        let q = self.hint(kernel::i128_to_fp(q_value));
        let r = self.gate_cell(a_masked.value - b_nonzero.value * q.value, |gate, ctx| {
            gate.sub_mul(ctx, a_masked.assigned, b_nonzero.assigned, q.assigned)
        });
        self.range_check(&q, bits);
        self.range_check(&r, bits);
        let gap = self.gate_cell(b_nonzero.value - Fp::one() - r.value, |gate, ctx| {
            let b_minus_r = gate.sub(ctx, b_nonzero.assigned, r.assigned);
            gate.dec(ctx, b_minus_r)
        });
        self.range_check(&gap, bits);
        Ok(q)
    }

    /// Python remainder `a % b` (sign of the divisor).
    fn mod_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Ok(self.div_rem(a, b, Rounding::Floor)?.1)
    }

    /// `base ** exp`. A constant exponent is unrolled by square-and-multiply
    /// (a negative one inverts the result, as the mock does). A dynamic
    /// exponent is decomposed into 6 bits, which constrains it to `[0, 64)`
    /// (larger or negative exponents have no witness), and multiplies the
    /// selected squares; the circuit shape does not depend on its value.
    fn pow_i(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        const DYNAMIC_EXP_BITS: usize = 6;
        if self.is_const_cell(exp) {
            let e = kernel::fp_to_i64(exp.value);
            let power = self.pow_const(base, e.unsigned_abs())?;
            return if e < 0 { self.inv_i(&power) } else { Ok(power) };
        }
        let bits = self.gate.num_to_bits(self.builder.main(0), exp.assigned, DYNAMIC_EXP_BITS);
        let one = self.constant_int(1)?;
        let mut acc = one;
        let mut square = *base;
        for (i, assigned) in bits.into_iter().enumerate() {
            if i > 0 {
                square = self.mul_i(&square, &square)?;
            }
            let value = *assigned.value();
            let bit = self.wrap(assigned, value);
            self.mark_bit(&bit);
            let factor = self.select_bit(&bit, &square, &one)?;
            acc = self.mul_i(&acc, &factor)?;
        }
        Ok(acc)
    }

    fn abs_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Ok(self.sign_and_abs(a)?.1)
    }

    /// `sign(a) = [a != 0] - 2 * [a < 0]`, from the pinned sign bit of
    /// `sign_and_abs`.
    fn sign_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let (negative, _) = self.sign_and_abs(a)?;
        let is_zero = self.is_zero_bit(a);
        let nonzero = self.logical_not(&is_zero)?;
        let two = Fp::from(2u64);
        Ok(self.gate_cell(nonzero.value - two * negative.value, |gate, ctx| {
            gate.sub_mul(ctx, nonzero.assigned, negative.assigned, QuantumCell::Constant(two))
        }))
    }

    /// `1 / a` in the field (the IR's `InvI`); 0 for `a = 0`, as in the mock.
    fn inv_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let one = self.constant_int(1)?;
        self.div_i(&one, a)
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
        let (q, _) = kernel::fp_mul_rescale(a.value, b.value, self.params.precision_bits);
        let raw = self.mul_i(a, b)?;
        self.rescale(&raw, q)
    }

    fn product_sum_f(&mut self, pairs: &[(Self::CellRef, Self::CellRef)]) -> Result<Self::CellRef, ProvingError> {
        let mut raw = self.constant_int(0)?;
        for (a, b) in pairs {
            raw = self.gate_cell(raw.value + a.value * b.value, |gate, ctx| {
                gate.mul_add(ctx, a.assigned, b.assigned, raw.assigned)
            });
        }
        let values: Vec<(Fp, Fp)> = pairs.iter().map(|(a, b)| (a.value, b.value)).collect();
        let (q, _) = kernel::fp_product_sum_rescale(&values, self.params.precision_bits);
        self.rescale(&raw, q)
    }

    /// Fixed-point `a / b`: `(a * 2^P) / b` rounded by `Rounding::Euclid`,
    /// as `kernel::fp_div_prescale`; 0 for a zero divisor.
    fn div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let scale = kernel::scale_fp(self.params.precision_bits);
        let scaled = self.gate_cell(a.value * scale, |gate, ctx| {
            gate.mul(ctx, a.assigned, QuantumCell::Constant(scale))
        });
        Ok(self.div_rem(&scaled, b, Rounding::Euclid)?.0)
    }

    /// Float `a // b = floor(a / b)` as a Q value. The ratio of two Q values
    /// is the ratio of their raw integers, so this is the integer floor
    /// division of the raw values, re-encoded.
    fn floor_div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let (quotient, _) = self.div_rem(a, b, Rounding::Floor)?;
        self.float_cast(&quotient)
    }

    /// Float `a % b` (sign of the divisor): the integer remainder of the raw
    /// Q values, which is already a Q value.
    fn mod_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Ok(self.div_rem(a, b, Rounding::Floor)?.1)
    }
    fn pow_f(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.pow_f_int(base, exp)
    }
    fn pow_f_int(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let max_exp = 8usize;
        let range = self.range_chip();
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
        let range = self.range_chip();
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

    /// `sin` by the algorithm of `kernel::fp_sin`: range reduction modulo
    /// `2 * pi_q`, fold onto `[0, pi_q)`, mirror onto `[0, pi_q / 2]`, Remez
    /// polynomial, sign.
    ///
    /// Soundness: `sign_and_abs`, the division by the constant `2 * pi_q`
    /// and every `mul_f` of the polynomial are functions of their inputs
    /// (see each). The two comparisons take operands already proven to lie
    /// in `[0, 2 * pi_q)`, inside the comparison's bit width. The rest are
    /// gates, so the output is a function of `a`.
    fn sin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let pi = kernel::pi_q(precision_bits);
        let compare_bits = self.lookup_aligned((128 - (2 * pi).leading_zeros()) as usize);
        let (negative, magnitude) = self.sign_and_abs(a)?;
        let (_, r) = self.div_rem_nonzero(&magnitude, Divisor::Constant(2 * pi), Rounding::Euclid)?;

        let below_pi = self.below_constant_bit(&r, pi, compare_bits);
        let folded_bit = self.logical_not(&below_pi)?;
        let pi_fp = kernel::i128_to_fp(pi);
        let folded = self.gate_cell(r.value - folded_bit.value * pi_fp, |gate, ctx| {
            gate.sub_mul(ctx, r.assigned, folded_bit.assigned, QuantumCell::Constant(pi_fp))
        });
        let in_first_quarter = self.below_constant_bit(&folded, pi / 2 + 1, compare_bits);
        let mirrored = self.gate_cell(pi_fp - folded.value, |gate, ctx| {
            gate.sub(ctx, QuantumCell::Constant(pi_fp), folded.assigned)
        });
        let argument = self.select_bit(&in_first_quarter, &folded, &mirrored)?;

        let poly = self.poly_eval_horner(&argument, &kernel::sin_coefs_q(precision_bits))?;
        let flip_value = negative.value + folded_bit.value - Fp::from(2u64) * negative.value * folded_bit.value;
        let flip = self.gate_cell(flip_value, |gate, ctx| gate.xor(ctx, negative.assigned, folded_bit.assigned));
        self.mark_bit(&flip);
        self.negate_if(&poly, &flip)
    }
    fn sinh_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("sinh_f") }
    /// `cos(a) = sin(a + pi/2)`, as `kernel::fp_cos`.
    fn cos_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let half_pi = kernel::i128_to_fp(kernel::half_pi_q(self.params.precision_bits));
        let shifted = self.gate_cell(a.value + half_pi, |gate, ctx| {
            gate.add(ctx, a.assigned, QuantumCell::Constant(half_pi))
        });
        self.sin_f(&shifted)
    }
    fn cosh_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("cosh_f") }
    fn tan_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("tan_f") }
    fn tanh_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("tanh_f") }
    /// `sqrt` as `kernel::fp_sqrt`: `floor(sqrt(a * 2^P))`, and 0 for a
    /// negative input. The pinned sign bit of `sign_and_abs` zeroes a
    /// negative input; `sqrt_nonneg` pins the root.
    fn sqrt_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let (negative, _) = self.sign_and_abs(a)?;
        let clamped = self.gate_cell((Fp::one() - negative.value) * a.value, |gate, ctx| {
            gate.mul_not(ctx, negative.assigned, a.assigned)
        });
        self.sqrt_nonneg(&clamped)
    }
    fn exp_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("exp_f") }
    fn log_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("log_f") }
    /// `arcsin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))`, as `kernel::fp_arcsin`
    /// (0 outside `[-1, 1]`).
    ///
    /// Soundness: every step is a function of its inputs. The domain bit
    /// compares the pinned magnitude `|x| <= 2^64` against `1` within 65
    /// bits. With `x'` = x inside the domain and 0 outside, `x'^2` (a pinned
    /// `mul_f`) lies in `[0, 1]`, so `1 - x'^2` meets `sqrt_nonneg`'s
    /// precondition; the divisor `1 + sqrt(..)` is positive, and the
    /// remainder bound of `div_rem_nonzero` re-proves it.
    fn arcsin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let one = kernel::scale_fp(precision_bits);
        let (_, magnitude) = self.sign_and_abs(a)?;
        let in_domain = self.below_constant_bit(&magnitude, (1i128 << precision_bits) + 1, 65);
        let x = self.mul_i(&in_domain, a)?;

        let x_squared = self.mul_f(&x, &x)?;
        let one_minus_x_squared = self.gate_cell(one - x_squared.value, |gate, ctx| {
            gate.sub(ctx, QuantumCell::Constant(one), x_squared.assigned)
        });
        let root = self.sqrt_nonneg(&one_minus_x_squared)?;
        let denominator = self.gate_cell(root.value + one, |gate, ctx| {
            gate.add(ctx, root.assigned, QuantumCell::Constant(one))
        });
        let scaled = self.gate_cell(x.value * one, |gate, ctx| {
            gate.mul(ctx, x.assigned, QuantumCell::Constant(one))
        });
        let (t, _) = self.div_rem_nonzero(&scaled, Divisor::Positive(denominator), Rounding::Euclid)?;

        let t_squared = self.mul_f(&t, &t)?;
        let odd_part = self.poly_eval_horner(&t_squared, &kernel::atan_odd_coefs_q(precision_bits))?;
        let half = self.mul_f(&t, &odd_part)?;
        self.add_i(&half, &half)
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

    /// Float→int, truncation toward zero (Python's `int()`), as
    /// `kernel::fp_int_cast`: the division of the raw Q value by `2^P` with
    /// the remainder taking the dividend's sign. `div_rem_nonzero` pins the
    /// quotient (including its signed 64-bit window, which the earlier
    /// lowering lacked: any remainder magnitude below `2^P` was accepted
    /// with a matching non-integer quotient).
    fn int_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let scale = 1i128 << self.params.precision_bits;
        Ok(self.div_rem_nonzero(a, Divisor::Constant(scale), Rounding::Trunc)?.0)
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
        // assert_const pins the cell to 1; an unsatisfiable witness (e.g.
        // `assert(False)`) would otherwise create_proof a verify-rejected
        // proof. Catch it during synthesis instead.
        if a.value != Fp::one() {
            let v = a.value;
            self.note_unsat(move || {
                format!(
                    "Witness does not satisfy circuit constraints: assert gate expected 1 but witness is {}",
                    kernel::fp_to_i64(v)
                )
            });
        }
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
            let range = self.range_chip();
            range.check_less_than_safe(self.ctx(), idx.assigned, hi);
            let decomposed = self.gate.num_to_bits(self.builder.main(0), idx.assigned, bits as usize);
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
        let range = self.range_chip();
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
        if a.value != Fp::zero() && a.value != Fp::one() {
            let v = a.value;
            self.note_unsat(move || {
                format!(
                    "Witness does not satisfy circuit constraints: bit gate expected 0 or 1 but witness is {}",
                    kernel::fp_to_i64(v)
                )
            });
        }
        self.gate.assert_bit(self.builder.main(0), a.assigned);
        self.mark_bit(a);
        Ok(())
    }

    fn allocate_memory(&mut self, segment_id: u32, size: u32, init: i64) -> Result<(), ProvingError> {
        let slots_version = self.segments.get(&segment_id).map_or(0, |s| s.slots_version + 1);
        self.segments.insert(
            segment_id,
            Segment {
                init: kernel::i64_to_fp(init),
                slots: vec![None; size as usize],
                overlay: Vec::new(),
                slots_version,
            },
        );
        Ok(())
    }
    /// See [`Segment`] for the program-order rule. A constant address must
    /// be in bounds at synthesis time; a dynamic one is range-checked.
    fn write_memory(&mut self, segment_id: u32, addr: &Self::CellRef, value: &Self::CellRef) -> Result<(), ProvingError> {
        let len = self.segment(segment_id)?.slots.len();
        let const_addr = self.is_const_cell(addr).then(|| kernel::fp_to_i64(addr.value));
        match const_addr {
            Some(a) if a < 0 || a as usize >= len => {
                return Err(ProvingError::synthesis("memory write out of bounds"));
            }
            Some(_) => {}
            None => {
                let range = self.range_chip();
                range.check_less_than_safe(self.ctx(), addr.assigned, len as u64);
            }
        }
        let segment = self.segments.get_mut(&segment_id).expect("checked above");
        match const_addr {
            Some(a) if segment.overlay.is_empty() => {
                segment.slots[a as usize] = Some(*value);
                segment.slots_version += 1;
            }
            _ => segment.overlay.push((*addr, *value)),
        }
        Ok(())
    }
    fn read_memory(&mut self, segment_id: u32, addr: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let base = if self.is_const_cell(addr) {
            let a = kernel::fp_to_i64(addr.value);
            let idx = usize::try_from(a).unwrap_or(usize::MAX);
            self.slot_cell(segment_id, idx)?
        } else {
            self.gather_slots(segment_id, addr)?
        };
        self.apply_overlay(segment_id, addr, base)
    }
    fn memory_trace_emit(&mut self, _: u32, _: bool, _: &[Self::CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError> { Ok(()) }

    fn allocate_dynamic_ndarray_meta(&mut self, array_id: u32, _: &str, max_length: u32, max_rank: u32, max_shape: &[u32]) -> Result<(), ProvingError> {
        let axis_bounds = if max_shape.is_empty() { vec![max_length; max_rank as usize] } else { max_shape.to_vec() };
        self.dyn_meta_bounds.insert(array_id, (max_length, axis_bounds));
        self.allocate_memory(10000 + array_id, max_length, 0)
    }
    fn witness_dynamic_ndarray_meta(&mut self, _: u32, _: u32, _: &str, _: &[Self::CellRef]) -> Result<(), ProvingError> { Ok(()) }
    fn dyn_meta_axis_bounds(&self, array_id: u32, max_rank: u32, max_length: u32) -> (u32, Vec<u32>) {
        self.dyn_meta_bounds
            .get(&array_id)
            .cloned()
            .unwrap_or_else(|| (max_length, vec![max_length; max_rank as usize]))
    }
    // assert_dynamic_ndarray_meta uses the trait-default lowering (was a no-op).
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

/// Test-only hint perturbation: the tamper tests replay a synthesis with
/// some hints changed, playing a prover that controls every hint.
#[cfg(test)]
mod tamper {
    use super::Fp;

    #[derive(Default)]
    pub(super) struct HintTamper {
        seen: usize,
        /// `(index, delta)` pairs: add `delta` to the `index`-th hint.
        targets: Vec<(usize, Fp)>,
    }

    impl HintTamper {
        pub(super) fn targeting(targets: &[(usize, Fp)]) -> Self {
            Self { seen: 0, targets: targets.to_vec() }
        }

        pub(super) fn apply(&mut self, value: Fp) -> Fp {
            let index = self.seen;
            self.seen += 1;
            self.targets
                .iter()
                .filter(|(i, _)| *i == index)
                .fold(value, |v, (_, delta)| v + delta)
        }

        pub(super) fn hints_seen(&self) -> usize {
            self.seen
        }
    }
}

#[cfg(test)]
#[path = "synthesizer_tests.rs"]
mod tests;
