//! Memory segments: constant-address wiring, dynamic-address reads by
//! one-hot selection, and the program-order overlay of dynamic writes.
//!
//! A dynamic read address is viewed through the provenance table's affine
//! form `addr = Σ root_i · scale_i + offset` (at most two roots, enforced by
//! the gates that built the address). Each root gets equality indicators
//! `[root == v]` for the values `v` that reach an in-bounds slot, shared by
//! every read on that root, and one membership assertion `Σ_v [root == v] = 1`
//! per value set. Under that assertion exactly one indicator is 1, so the
//! read is the inner product `Σ_v [root == v] · slot(v)`: three cells per
//! choice, no selection chain.

use std::collections::HashMap;

use halo2_base::gates::GateInstructions;
use halo2_base::QuantumCell;

use super::ranges::{Interval, Operand};
use super::{Fp, Halo2BaseCell, Halo2BaseSynthesizer};
use crate::prove::error::ProvingError;
use crate::prove::kernel;
use crate::prove::traits::Synthesizer;

/// The most `[write addr == read addr]` tests the dynamic-write overlay may
/// emit in one circuit. Each costs about 16 cells (equality and selection),
/// so the budget is about 2^24 cells, a circuit that needs more memory to
/// prove than the benchmark harness allows a step (8 GB). A program over it
/// (for example a scatter of 20,000 values to runtime addresses read back
/// 20,000 times: 4·10^8 tests, which exhausted 16 GB during synthesis) is
/// rejected with an error naming the construct. The overlay is quadratic by
/// construction; such programs need a memory-consistency argument, which
/// this backend does not have.
const OVERLAY_TEST_BUDGET: usize = 1 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum OverlayReadKey {
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
pub(super) struct Segment {
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

/// Affine view of an integer cell: `cell == Σ root_i * scale_i + offset`,
/// with at most two dynamic roots and non-zero scales.
#[derive(Clone, Debug, Default)]
struct AffineLin {
    terms: Vec<(Halo2BaseCell, i64)>,
    offset: i64,
}

/// A value a read may select: a cell, or a slot never written (the
/// segment's constant init value).
#[derive(Clone, Copy)]
enum SlotValue {
    Cell(Halo2BaseCell),
    Init(Fp),
}

/// Memory state of one synthesis.
#[derive(Default)]
pub(super) struct MemoryState {
    segments: HashMap<u32, Segment>,
    /// `[root == v]` indicator bits, shared by every read on the same root.
    indicators: HashMap<(u64, i64), Halo2BaseCell>,
    /// `(root id, value set)` memberships already asserted.
    memberships: std::collections::HashSet<(u64, Vec<i64>)>,
    /// Stage-1 planes of two-root reads, keyed by (segment, slots version,
    /// root id, root scale, residual offset).
    planes: HashMap<(u32, u64, u64, i64, i64), Halo2BaseCell>,
    /// `[write addr == read addr]` bits for overlay writes; the read side is
    /// keyed by cell id for dynamic addresses and by value for constants.
    overlay_hits: HashMap<(u64, OverlayReadKey), Halo2BaseCell>,
    /// Overlay equality tests emitted so far (see `OVERLAY_TEST_BUDGET`).
    overlay_tests: usize,
    /// A lower budget for tests.
    #[cfg(test)]
    pub(super) overlay_budget: Option<usize>,
}

impl Halo2BaseSynthesizer<'_> {
    pub(super) fn memory_allocate(&mut self, segment_id: u32, size: u32, init: i64) {
        let slots_version = self.memory.segments.get(&segment_id).map_or(0, |s| s.slots_version + 1);
        let segment = Segment {
            init: kernel::i64_to_fp(init),
            slots: vec![None; size as usize],
            overlay: Vec::new(),
            slots_version,
        };
        self.memory.segments.insert(segment_id, segment);
    }

    /// See [`Segment`] for the program-order rule. A constant address must
    /// be in bounds at synthesis time; a dynamic one is asserted below the
    /// segment length.
    pub(super) fn memory_write(
        &mut self,
        segment_id: u32,
        addr: &Halo2BaseCell,
        value: &Halo2BaseCell,
    ) -> Result<(), ProvingError> {
        let len = self.segment(segment_id)?.slots.len();
        let const_addr = self.is_const_cell(addr).then(|| kernel::fp_to_i64(addr.value));
        match const_addr {
            Some(a) if a < 0 || a as usize >= len => {
                return Err(ProvingError::synthesis("memory write out of bounds"));
            }
            Some(_) => {}
            None => self.assert_less(addr, Operand::Constant(len as i128))?,
        }
        let segment = self.memory.segments.get_mut(&segment_id).expect("checked above");
        match const_addr {
            Some(a) if segment.overlay.is_empty() => {
                segment.slots[a as usize] = Some(*value);
                segment.slots_version += 1;
            }
            _ => segment.overlay.push((*addr, *value)),
        }
        Ok(())
    }

    pub(super) fn memory_read(&mut self, segment_id: u32, addr: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let base = if self.is_const_cell(addr) {
            let idx = usize::try_from(kernel::fp_to_i64(addr.value)).unwrap_or(usize::MAX);
            self.slot_cell(segment_id, idx)?
        } else {
            self.gather_slots(segment_id, addr)?
        };
        self.apply_overlay(segment_id, addr, base)
    }

    fn segment(&self, segment_id: u32) -> Result<&Segment, ProvingError> {
        self.memory
            .segments
            .get(&segment_id)
            .ok_or_else(|| ProvingError::synthesis("segment not allocated"))
    }

    /// What a slot holds before the first dynamic write: its last
    /// constant-address write, or the segment's init value. Addresses past
    /// the end read the init value, as in the mock backend.
    fn slot_value(&self, segment_id: u32, idx: usize) -> Result<SlotValue, ProvingError> {
        let segment = self.segment(segment_id)?;
        Ok(match segment.slots.get(idx).copied().flatten() {
            Some(cell) => SlotValue::Cell(cell),
            None => SlotValue::Init(segment.init),
        })
    }

    /// [`Self::slot_value`] as a cell (the init value as a fixed constant; a
    /// witness cell here would be a free variable).
    fn slot_cell(&mut self, segment_id: u32, idx: usize) -> Result<Halo2BaseCell, ProvingError> {
        Ok(match self.slot_value(segment_id, idx)? {
            SlotValue::Cell(cell) => cell,
            SlotValue::Init(init) => self.const_cell(init),
        })
    }

    /// Affine view of `cell`, read live from the provenance table (terms
    /// translate root ValueIds to their cells; a root without a cell yet
    /// degrades to the identity view). Falls back to the identity view
    /// (the cell is its own root) for dynamic cells, and to a pure offset
    /// for constants.
    fn affine_lin_of(&self, cell: &Halo2BaseCell) -> AffineLin {
        // A constant is its own value (constant cells are shared between IR
        // values, so the provenance of whichever value took it last is moot).
        if self.is_const_cell(cell) {
            return AffineLin { terms: Vec::new(), offset: kernel::fp_to_i64(cell.value) };
        }
        let provenance = self
            .vid_of_cell
            .get(&cell.id)
            .and_then(|vid| self.prov.affine_of(*vid))
            .filter(|view| !view.terms.is_empty());
        if let Some(view) = provenance {
            let terms: Option<Vec<(Halo2BaseCell, i64)>> = view
                .terms
                .iter()
                .map(|(root, scale)| self.cell_of_vid.get(root).map(|c| (*c, *scale)))
                .collect();
            if let Some(terms) = terms {
                return AffineLin { terms, offset: view.offset };
            }
        }
        AffineLin { terms: vec![(*cell, 1)], offset: 0 }
    }

    /// `[root == v]`, shared by every read on the same root.
    fn root_indicator(&mut self, root: &Halo2BaseCell, v: i64) -> Result<Halo2BaseCell, ProvingError> {
        if let Some(ind) = self.memory.indicators.get(&(root.id, v)) {
            return Ok(*ind);
        }
        let v_cell = self.const_cell(kernel::i64_to_fp(v));
        let ind = self.eq_i(root, &v_cell)?;
        self.memory.indicators.insert((root.id, v), ind);
        Ok(ind)
    }

    /// Assert (once per root and value set) that exactly one indicator
    /// `[root == v]`, `v` in `vs`, is 1: the root is one of the values. The
    /// indicators are bits by their gadget, and distinct values cannot both
    /// equal the root, so the sum is 1 exactly when the root is in `vs`.
    fn assert_one_hot(&mut self, root: &Halo2BaseCell, vs: &[i64]) -> Result<(), ProvingError> {
        let key = (root.id, vs.to_vec());
        if self.memory.memberships.contains(&key) {
            return Ok(());
        }
        let indicators = vs
            .iter()
            .map(|v| self.root_indicator(root, *v))
            .collect::<Result<Vec<_>, _>>()?;
        let total: Fp = indicators.iter().map(|ind| ind.value).sum();
        let sum = self.gate_cell(total, |gate, ctx| {
            gate.sum(ctx, indicators.iter().map(|ind| QuantumCell::Existing(ind.assigned)))
        });
        self.assert_const(&sum, Fp::one());
        self.memory.memberships.insert(key);
        if let (Some(lo), Some(hi)) = (vs.iter().min(), vs.iter().max()) {
            if let Some(range) = Interval::new(*lo as i128, *hi as i128) {
                self.prove_range(root, range);
            }
        }
        Ok(())
    }

    /// [`Self::one_hot_sum`] over cell values.
    pub(super) fn one_hot_combination(&mut self, choices: &[(Halo2BaseCell, Halo2BaseCell)]) -> Halo2BaseCell {
        let choices: Vec<(Halo2BaseCell, SlotValue)> =
            choices.iter().map(|(ind, value)| (*ind, SlotValue::Cell(*value))).collect();
        self.one_hot_sum(&choices)
    }

    /// `Σ indicator · value` for indicators of which the caller has asserted
    /// that exactly one is 1 (a one-hot membership): the selected value.
    /// Choices holding the constant 0 contribute nothing and are dropped.
    fn one_hot_sum(&mut self, choices: &[(Halo2BaseCell, SlotValue)]) -> Halo2BaseCell {
        let terms: Vec<(Halo2BaseCell, QuantumCell<Fp>, Fp)> = choices
            .iter()
            .filter_map(|(ind, value)| match value {
                SlotValue::Cell(cell) => Some((*ind, QuantumCell::Existing(cell.assigned), cell.value)),
                SlotValue::Init(v) if *v == Fp::zero() => None,
                SlotValue::Init(v) => Some((*ind, QuantumCell::Constant(*v), *v)),
            })
            .collect();
        if terms.is_empty() {
            return self.const_cell(Fp::zero());
        }
        let value: Fp = terms.iter().map(|(ind, _, v)| ind.value * v).sum();
        self.gate_cell(value, |gate, ctx| {
            gate.inner_product(
                ctx,
                terms.iter().map(|(_, q, _)| *q),
                terms.iter().map(|(ind, _, _)| QuantumCell::Existing(ind.assigned)),
            )
        })
    }

    /// The values a root may take for the read: those whose slot index
    /// `offset + scale·v` is in `[0, len)`, narrowed by the root's interval
    /// (provenance or proven range; see `interval_of`).
    fn root_choices(&self, root: &Halo2BaseCell, scale: i64, offset: i64, len: usize) -> Vec<(i64, usize)> {
        let interval = self.interval_of(root);
        (0..len as i64)
            .filter_map(|idx| {
                let num = idx - offset;
                (scale != 0 && num % scale == 0).then(|| (num / scale, idx as usize))
            })
            .filter(|(v, _)| interval.is_none_or(|iv| iv.lo <= *v as i128 && (*v as i128) <= iv.hi))
            .collect()
    }

    /// The slot value at a dynamic address, before overlay writes.
    ///
    /// One root (`addr = root·s + c`): the inner product of the root's
    /// indicators with the reachable slots, under the root's membership
    /// assertion over those values, which also proves the address in
    /// bounds.
    ///
    /// Two roots (`addr = r1·s1 + r2·s2 + c` with `s2 | s1`, a windowed
    /// gather): stage-1 planes `Σ_v [r1 == v] · mem[v·s1 + t]` per residual
    /// `t = w·s2 + c`, cached and shared across reads, then the inner product
    /// of `r2`'s indicators with the planes. Both roots are asserted one-hot
    /// and the full address is asserted below the segment length: a pair
    /// `(v, w)` with an out-of-bounds index would otherwise read 0.
    fn gather_slots(&mut self, segment_id: u32, addr: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let len = self.segment(segment_id)?.slots.len();
        let lin = self.affine_lin_of(addr);
        if lin.terms.len() == 2 {
            if let Some(value) = self.gather_two_roots(segment_id, addr, &lin, len)? {
                return Ok(value);
            }
        }
        let (root, scale, offset) = match lin.terms.len() {
            1 => (lin.terms[0].0, lin.terms[0].1, lin.offset),
            _ => (*addr, 1, 0),
        };
        let choices = self.root_choices(&root, scale, offset, len);
        if choices.is_empty() {
            return Err(ProvingError::synthesis("dynamic memory read can never be in bounds"));
        }
        let vs: Vec<i64> = choices.iter().map(|(v, _)| *v).collect();
        self.assert_one_hot(&root, &vs)?;
        let mut selected = Vec::with_capacity(choices.len());
        for (v, idx) in &choices {
            selected.push((self.root_indicator(&root, *v)?, self.slot_value(segment_id, *idx)?));
        }
        Ok(self.one_hot_sum(&selected))
    }

    /// The two-root gather of [`Self::gather_slots`]; `None` when the view
    /// does not have the digit structure it needs.
    fn gather_two_roots(
        &mut self,
        segment_id: u32,
        addr: &Halo2BaseCell,
        lin: &AffineLin,
        len: usize,
    ) -> Result<Option<Halo2BaseCell>, ProvingError> {
        let (mut major, mut minor) = (lin.terms[0], lin.terms[1]);
        if major.1.abs() < minor.1.abs() {
            std::mem::swap(&mut major, &mut minor);
        }
        let ((root1, s1), (root2, s2)) = (major, minor);
        if !(s1 > 0 && s2 > 0 && s1 % s2 == 0 && s1 / s2 >= 2) {
            return Ok(None);
        }
        // Digit-canonical choices for the minor root: residuals inside one
        // major stride (Zinnia's clamped in-bounds index lowering). A kernel
        // violating it fails the membership assertion, it does not prove.
        let minor_interval = self.interval_of(&root2);
        let ws: Vec<i64> = (0..s1 / s2)
            .filter(|w| minor_interval.is_none_or(|iv| iv.lo <= *w as i128 && (*w as i128) <= iv.hi))
            .filter(|w| !self.root_choices(&root1, s1, w * s2 + lin.offset, len).is_empty())
            .collect();
        let mut vs: Vec<i64> = ws
            .iter()
            .flat_map(|w| self.root_choices(&root1, s1, w * s2 + lin.offset, len))
            .map(|(v, _)| v)
            .collect();
        vs.sort_unstable();
        vs.dedup();
        if ws.is_empty() || vs.is_empty() {
            return Ok(None);
        }
        self.assert_less(addr, Operand::Constant(len as i128))?;
        self.assert_one_hot(&root2, &ws)?;
        self.assert_one_hot(&root1, &vs)?;
        let mut planes = Vec::with_capacity(ws.len());
        for w in &ws {
            let plane = self.gathered_plane(segment_id, &root1, s1, w * s2 + lin.offset, len)?;
            planes.push((self.root_indicator(&root2, *w)?, SlotValue::Cell(plane)));
        }
        Ok(Some(self.one_hot_sum(&planes)))
    }

    /// `Σ_v [root == v] · mem[v·scale + t]` over the root's in-bounds
    /// choices, cached per (segment, slots version, root, scale, t). Only
    /// meaningful under the root's membership assertion and the read's
    /// address bound (an out-of-bounds pair would read 0 here).
    fn gathered_plane(
        &mut self,
        segment_id: u32,
        root: &Halo2BaseCell,
        scale: i64,
        t: i64,
        len: usize,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let version = self.segment(segment_id)?.slots_version;
        let key = (segment_id, version, root.id, scale, t);
        if let Some(plane) = self.memory.planes.get(&key) {
            return Ok(*plane);
        }
        let mut selected = Vec::new();
        for (v, idx) in self.root_choices(root, scale, t, len) {
            selected.push((self.root_indicator(root, v)?, self.slot_value(segment_id, idx)?));
        }
        let plane = self.one_hot_sum(&selected);
        self.memory.planes.insert(key, plane);
        Ok(plane)
    }

    /// Whether an overlay write at `write_addr` is seen by a read at
    /// `read_addr`. Two constant addresses are compared at synthesis time;
    /// otherwise an equality bit, cached so repeated reads against the same
    /// pending write reuse it.
    fn overlay_hit(&mut self, write_addr: &Halo2BaseCell, read_addr: &Halo2BaseCell) -> Result<OverlayHit, ProvingError> {
        if self.is_const_cell(write_addr) && self.is_const_cell(read_addr) {
            return Ok(if write_addr.value == read_addr.value { OverlayHit::Always } else { OverlayHit::Never });
        }
        let key = if self.is_const_cell(read_addr) {
            OverlayReadKey::Const(kernel::fp_to_i64(read_addr.value))
        } else {
            OverlayReadKey::Cell(read_addr.id)
        };
        if let Some(hit) = self.memory.overlay_hits.get(&(write_addr.id, key)) {
            return Ok(OverlayHit::When(*hit));
        }
        let hit = self.eq_i(write_addr, read_addr)?;
        self.memory.overlay_hits.insert((write_addr.id, key), hit);
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
        self.memory.overlay_tests += overlay.len();
        #[cfg(test)]
        let budget = self.memory.overlay_budget.unwrap_or(OVERLAY_TEST_BUDGET);
        #[cfg(not(test))]
        let budget = OVERLAY_TEST_BUDGET;
        if self.memory.overlay_tests > budget {
            return Err(ProvingError::synthesis(format!(
                "dynamic-address memory writes: reads after {} runtime-address writes to one array need more \
                 than {} address comparisons in total; the halo2-base backend has no memory-consistency \
                 argument for scatters of this size",
                overlay.len(),
                budget
            )));
        }
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
}
