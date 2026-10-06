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

mod bitwise;
mod fixed_point;
mod identities;
mod keys;
mod memory;
mod plan;
mod ranges;

use self::fixed_point::Divisor;
use self::memory::MemoryState;
use self::plan::Plan;
use self::ranges::{Comparison, Interval, Operand, TwoValued};
use crate::circuit_input::{InputPath, ResolvedWitness};
use crate::prove::error::ProvingError;
use crate::prove::field::PrimeField;
use crate::prove::kernel::{self, Rounding};
use crate::prove::traits::Synthesizer;
use crate::prove::types::ProvingParams;

#[derive(Clone, Copy, Debug)]
pub struct Halo2BaseCell {
    pub id: u64,
    /// The advice cell holding the value; `None` for a constant that has no
    /// cell of its own (see [`Halo2BaseSynthesizer::const_cell`]). Gates take
    /// the cell through [`Halo2BaseCell::quantum`], gadgets that need an
    /// assigned cell through [`Halo2BaseSynthesizer::assigned`].
    assigned: Option<AssignedValue<Fp>>,
    pub value: Fp,
}

impl Halo2BaseCell {
    /// The value as a gate operand: its advice cell (a copy constraint to
    /// it), or for a constant without a cell the constant itself, which the
    /// gate's cell takes by a copy constraint to the constants column. Either
    /// way the gate's cell equals the value; the constant form spares the
    /// separate cell `load_constant` would add.
    fn quantum(&self) -> QuantumCell<Fp> {
        match self.assigned {
            Some(assigned) => QuantumCell::Existing(assigned),
            None => QuantumCell::Constant(self.value),
        }
    }
}

pub struct Halo2BaseSynthesizer<'a> {
    builder: &'a mut BaseCircuitBuilder<Fp>,
    /// Checks laid out after the verifier's challenges (`circuit.rs`),
    /// registered by the lowerings that need them.
    pub(crate) second_phase: super::circuit::SecondPhase,
    /// Read-only memory tables and the reads looked up in them
    /// (`IR::ReadTable`, `circuit/tables.rs`).
    pub(crate) tables: super::circuit::Tables,
    /// Read-write memory traces (`circuit/ram.rs`) of the segments the plan
    /// puts in RAM mode.
    pub(crate) ram: super::circuit::Ram,
    witness: Option<ResolvedWitness>,
    params: ProvingParams,
    public_values: Vec<Fp>,
    /// Memory segments and the caches of dynamic reads (see `memory.rs`).
    memory: MemoryState,
    /// Q values that are float casts `n·2^P` of an integer cell `n`, by the
    /// Q cell's id (see `fixed_point::Factor`).
    float_casts: HashMap<u64, Halo2BaseCell>,
    /// Cells known to be one of two constants by a bit (see `ranges.rs`).
    two_valued: HashMap<u64, TwoValued>,
    /// Comparison bits against a constant, by bit id, and negated bits (the
    /// `not` cell's id to its operand's), for `refine_by_assertion`.
    comparisons: HashMap<u64, Comparison>,
    negations: HashMap<u64, u64>,
    /// `a - b` cells by `(a.id, b.id)`, shared by comparisons and selects.
    differences: HashMap<(u64, u64), Halo2BaseCell>,
    /// Low two's-complement bits per cell id (see `bitwise.rs`).
    bit_cache: HashMap<u64, Vec<Halo2BaseCell>>,
    /// Packed sort keys `v·2^b + i` by `(v.id, i.id, b)` (see `keys.rs`).
    packed_keys: HashMap<(u64, u64, u32), Halo2BaseCell>,
    dyn_meta_bounds: HashMap<u32, (u32, Vec<u32>)>,
    next_id: u64,
    bit_cells: HashSet<halo2_base::ContextCell>,
    bit_cell_ids: HashSet<u64>,
    const_cell_ids: HashSet<u64>,
    /// One constant cell per value (the field element's bytes); equal
    /// constants are interchangeable. See [`Halo2BaseSynthesizer::const_cell`].
    constants: HashMap<[u8; 32], Halo2BaseCell>,
    /// The advice cell of each constant some gadget needed as a cell (see
    /// [`Halo2BaseSynthesizer::assigned`]), loaded once per value.
    loaded_constants: HashMap<[u8; 32], AssignedValue<Fp>>,
    select_many_cache: HashMap<Vec<u64>, Vec<Halo2BaseCell>>,
    u128_lt_bit_cache: HashMap<(u64, u64, u32), Vec<Halo2BaseCell>>,
    /// Constraint-proven integer ranges per cell id (see `ranges.rs`).
    /// Comparisons size their checks by them and skip window checks the
    /// constraints already imply.
    proven: HashMap<u64, Interval>,
    /// Pinned sign bits `[x < 0]` by cell id (sign tests, `sign_and_abs`).
    sign_bits: HashMap<u64, Halo2BaseCell>,
    /// The int64 window checks emitted so far, by the checked cell's id
    /// (see `ranges::WindowCheck`): a later sign test reads the top limb.
    windows: HashMap<u64, ranges::WindowCheck>,
    /// Signed-window order bits `[a < b]` of two cells, by `(a, b)` cell ids
    /// (see `ranges::less_than`: the reverse order then needs one equality).
    orders: HashMap<(u64, u64), Halo2BaseCell>,
    /// Cells an emitted constraint proves non-zero (an asserted `x != 0`).
    nonzero: HashSet<u64>,
    /// `(sign, magnitude)` hints of dynamic divisors, by cell id: pinned by
    /// the remainder bound of the division that made them (see
    /// `div_rem_nonzero`), so later divisions by the same cell reuse them.
    divisor_signs: HashMap<u64, (Halo2BaseCell, Halo2BaseCell)>,
    // (sign bit, magnitude) per cell id, see `sign_and_abs`.
    sign_abs_cache: HashMap<u64, (Halo2BaseCell, Halo2BaseCell)>,
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
    /// Facts read off the whole IR graph ahead of synthesis (see `plan.rs`);
    /// empty unless [`Halo2BaseSynthesizer::plan_ahead`] is called.
    plan: Plan,
    /// The statement being lowered (from `on_stmt_begin`).
    current_stmt: u32,
    /// Deferred products by placeholder id: the factors of a `MulI` that the
    /// plan fuses into its one consumer (see `mul_i`).
    deferred: HashMap<u64, (Halo2BaseCell, Halo2BaseCell)>,
    /// Address values the plan keeps without a cell (`Plan::address_only`),
    /// by placeholder id; the cell an access computed for one, once it did.
    address_placeholders: HashMap<u64, Option<Halo2BaseCell>>,
    /// `(cell id, advice length)` right after `gate_from` emitted the cell:
    /// while the length is unchanged, that cell is the context's last one.
    last_output: Option<(u64, usize)>,
    /// Per-statement cell counts, recorded only when enabled (see
    /// [`Halo2BaseSynthesizer::enable_profile`]); a measurement aid.
    profile: Option<CellProfile>,
    /// Test-only hook that perturbs prover hints (see `hint`).
    #[cfg(test)]
    tamper: tamper::HintTamper,
}

/// Advice and lookup cells emitted per IR statement (a measurement aid for
/// `cell_stats`; never enabled on the proving path).
#[derive(Default)]
pub struct CellProfile {
    current: Option<(u32, usize, usize)>,
    /// `(statement id, advice cells, lookup cells)`.
    pub per_stmt: Vec<(u32, usize, usize)>,
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
            second_phase: super::circuit::SecondPhase::default(),
            tables: super::circuit::Tables::default(),
            ram: super::circuit::Ram::default(),
            witness,
            params,
            public_values: Vec::new(),
            memory: MemoryState::default(),
            float_casts: HashMap::new(),
            two_valued: HashMap::new(),
            bit_cache: HashMap::new(),
            packed_keys: HashMap::new(),
            differences: HashMap::new(),
            comparisons: HashMap::new(),
            sign_bits: HashMap::new(),
            windows: HashMap::new(),
            orders: HashMap::new(),
            nonzero: HashSet::new(),
            divisor_signs: HashMap::new(),
            negations: HashMap::new(),
            dyn_meta_bounds: HashMap::new(),
            next_id: 0,
            bit_cells: HashSet::new(),
            bit_cell_ids: HashSet::new(),
            const_cell_ids: HashSet::new(),
            constants: HashMap::new(),
            loaded_constants: HashMap::new(),
            select_many_cache: HashMap::new(),
            u128_lt_bit_cache: HashMap::new(),
            proven: HashMap::new(),
            sign_abs_cache: HashMap::new(),
            prov: provenance,
            vid_of_cell: HashMap::new(),
            cell_of_vid: HashMap::new(),
            first_unsat: None,
            gate: GateChip::<Fp>::new(),
            range: None,
            plan: Plan::default(),
            current_stmt: 0,
            deferred: HashMap::new(),
            address_placeholders: HashMap::new(),
            last_output: None,
            profile: None,
            #[cfg(test)]
            tamper: tamper::HintTamper::default(),
        }
    }

    /// Read the facts of `ir` the lowering uses ahead of the statements that
    /// establish them (planned bounds, deferred products; see `plan.rs`).
    pub fn plan_ahead(&mut self, ir: &crate::ir::IRGraph) {
        self.plan = Plan::with_provenance(ir, self.prov);
    }

    /// Record the advice and lookup cells each statement emits (costly:
    /// counting lookups walks every queued lookup cell).
    pub fn enable_profile(&mut self) {
        self.profile = Some(CellProfile::default());
    }

    /// The recorded profile, closing the last statement's entry.
    pub fn take_profile(&mut self) -> Option<CellProfile> {
        self.close_profile_entry();
        self.profile.take()
    }

    fn cell_counters(&mut self) -> (usize, usize) {
        let stats = self.builder.statistics();
        (stats.gate.total_advice_per_phase[0], stats.total_lookup_advice_per_phase[0])
    }

    fn close_profile_entry(&mut self) {
        let Some((stmt, advice0, lookup0)) = self.profile.as_mut().and_then(|p| p.current.take()) else {
            return;
        };
        let (advice, lookup) = self.cell_counters();
        if let Some(profile) = self.profile.as_mut() {
            profile.per_stmt.push((stmt, advice - advice0, lookup - lookup0));
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
        let value = self.hint_value(value);
        self.cell(value)
    }

    /// The value a hint cell receives: `value`, except under the tamper
    /// tests, which replace it. For hints laid out by a gadget directly
    /// (`bitwise::hinted_bits`) rather than loaded by [`Self::hint`].
    fn hint_value(&mut self, value: Fp) -> Fp {
        #[cfg(test)]
        let value = self.tamper.apply(value);
        value
    }

    /// A hint the IR asserts to be a bit (`plan.bit_hints`), assigned inside
    /// its bit gate `0 + h·h = h`: the gate's three copies of `h` are its
    /// cells (copy-constrained equal), so the hint costs the gate's four
    /// cells instead of a free cell and a four-cell gate. Marked as a bit, so
    /// the `AssertBit` that the plan saw emits nothing; the gate is exactly
    /// the constraint that `AssertBit` emits.
    fn hint_in_bit_gate(&mut self, value: Fp) -> Halo2BaseCell {
        let value = self.hint_value(value);
        if value != Fp::zero() && value != Fp::one() {
            self.note_unsat(move || {
                format!(
                    "Witness does not satisfy circuit constraints: bit gate expected 0 or 1 but witness is {}",
                    kernel::fp_to_i64(value)
                )
            });
        }
        let ctx = self.builder.main(0);
        let w = QuantumCell::Witness(value);
        ctx.assign_region([QuantumCell::Constant(Fp::zero()), w, w, w], [0]);
        let (b, c, d) = (ctx.get(-3), ctx.get(-2), ctx.get(-1));
        ctx.constrain_equal(&b, &c);
        ctx.constrain_equal(&b, &d);
        let bit = self.wrap(b, value);
        self.mark_bit(&bit);
        bit
    }

    /// A boolean hint, constrained to {0, 1} and marked as a bit.
    fn hint_bit(&mut self, value: bool) -> Halo2BaseCell {
        let bit = self.hint(if value { Fp::one() } else { Fp::zero() });
        let bit_cell = self.assigned(&bit);
        self.gate.assert_bit(self.builder.main(0), bit_cell);
        self.mark_bit(&bit);
        bit
    }

    /// `bits` rounded up to a multiple of the lookup width: a range check
    /// of that many bits costs no more limbs and needs no extra cell for a
    /// partial top limb.
    fn lookup_aligned(&mut self, bits: usize) -> usize {
        let lookup = self.range_chip().lookup_bits();
        bits.div_ceil(lookup) * lookup
    }

    /// The constant `value` (one cell id per value, then shared). It gets no
    /// advice cell here: a gate takes it as a constant operand
    /// ([`Halo2BaseCell::quantum`]), and only a gadget that needs an assigned
    /// cell (a range check, a public output, an equality constraint) loads
    /// it, once per value ([`Self::assigned`]). Constants that only steer
    /// lowering decisions, such as the bounds of an input-domain check the
    /// window check already proves, then cost nothing.
    fn const_cell(&mut self, value: Fp) -> Halo2BaseCell {
        let key = value.to_repr();
        if let Some(cell) = self.constants.get(&key) {
            return *cell;
        }
        let id = self.next_id;
        self.next_id += 1;
        let cell = Halo2BaseCell { id, assigned: None, value };
        self.const_cell_ids.insert(cell.id);
        self.constants.insert(key, cell);
        cell
    }

    /// The advice cell of `x`, for gadgets that take an assigned cell
    /// rather than a gate operand. A constant without a cell is loaded with
    /// `load_constant` (fixed by a copy constraint to the constants column)
    /// the first time a gadget needs it.
    fn assigned(&mut self, x: &Halo2BaseCell) -> AssignedValue<Fp> {
        if let Some(assigned) = x.assigned {
            return assigned;
        }
        let key = x.value.to_repr();
        if let Some(assigned) = self.loaded_constants.get(&key) {
            return *assigned;
        }
        let assigned = self.ctx().load_constant(x.value);
        self.loaded_constants.insert(key, assigned);
        assigned
    }

    /// A private input `x` that an `AssertU128Lt(x, bound)` constrains to
    /// `[0, bound)` (`Plan::bounded_inputs`), for `bound <= 2^lookup_bits`:
    /// `x` is assigned inside that check's gate instead of a cell of its own,
    /// and the check is done here (the assertion later finds `x`'s range
    /// proven and emits nothing; it holds on every path, so checking at the
    /// read is the same constraint). `None` when the bound needs no gate.
    ///
    /// Soundness: `x` and the gate's other cell `t` are each looked up in
    /// `[0, 2^lb)`, so both are small integers and the gate is an integer
    /// identity. For `bound = 2^b < 2^lb`, `t = x·2^(lb-b) < 2^lb` gives
    /// `x < 2^b`; otherwise `t = bound - 1 - x >= 0` gives `x <= bound - 1`.
    fn read_bounded_input(&mut self, value: Fp, bound: i128) -> Option<Halo2BaseCell> {
        let lookup_bits = self.params.lookup_bits as usize;
        let full = 1i128 << lookup_bits;
        if bound <= 1 || bound >= full {
            return None;
        }
        if !kernel::fp_to_i128(value).is_some_and(|v| (0..bound).contains(&v)) {
            self.note_unsat(move || format!("Witness does not satisfy circuit constraints: an input is not below {bound}"));
        }
        let ctx = self.builder.main(0);
        let (x, t) = if (bound as u128).is_power_of_two() {
            let k = kernel::i128_to_fp(full / bound);
            ctx.assign_region(
                [QuantumCell::Constant(Fp::zero()), QuantumCell::Witness(value), QuantumCell::Constant(k), QuantumCell::Witness(value * k)],
                [0],
            );
            (ctx.get(-3), ctx.get(-1))
        } else {
            let limit = kernel::i128_to_fp(bound - 1);
            ctx.assign_region(
                [QuantumCell::Witness(limit - value), QuantumCell::Witness(value), QuantumCell::Constant(Fp::one()), QuantumCell::Constant(limit)],
                [0],
            );
            (ctx.get(-3), ctx.get(-4))
        };
        let x = self.wrap(x, value);
        let t_value = *t.value();
        let t = self.wrap(t, t_value);
        self.range_check(&x, lookup_bits);
        self.range_check(&t, lookup_bits);
        self.prove_range(&x, Interval { lo: 0, hi: bound - 1 });
        Some(x)
    }

    /// `assert x >= 0` (`x`, then the constant 0) when the plan found a later
    /// `sqrt(x)` (`Plan::root_guards`) and `x` is already proven inside the
    /// window: no check here, `x`'s range becomes `[0, hi]`, and `sqrt_f`
    /// then takes `sqrt_nonneg`, whose constraints `x·2^P = y^2 + d` with
    /// `y` and `d` range-checked far below `p` have no solution for a
    /// negative `x` (`|x·2^P| < 2^(63+P)`, so the identity is one of
    /// integers). The assertion therefore holds for every accepted witness;
    /// a circuit has no control flow, so the root's constraints apply
    /// wherever they are emitted. `None` leaves the assertion to
    /// `assert_relation`.
    fn assert_nonneg_by_root(&mut self, x: &Halo2BaseCell, zero: &Halo2BaseCell) -> Result<Option<Halo2BaseCell>, ProvingError> {
        if !self.plan.root_guards.contains(&self.current_stmt) || !self.is_const_cell(zero) || zero.value != Fp::zero() {
            return Ok(None);
        }
        let Some(range) = self.proven_range(x).filter(|r| r.within(&Interval::SIGNED64)) else { return Ok(None) };
        if kernel::fp_is_negative(x.value) {
            self.note_unsat(|| "Witness does not satisfy circuit constraints: an asserted comparison is false".to_string());
        }
        if let Some(nonneg) = Interval::new(range.lo.max(0), range.hi) {
            self.prove_range(x, nonneg);
        }
        Ok(Some(self.constant_bool(true)?))
    }

    /// Whether the statement being lowered is a comparison whose only use is
    /// `Assert` (see `plan.rs`).
    fn is_asserted_comparison(&self) -> bool {
        self.plan.asserted_comparisons.contains(&self.current_stmt)
    }

    /// The truth value of a constant cell, if `a` is one.
    fn constant_truth(&self, a: &Halo2BaseCell) -> Option<bool> {
        self.is_const_cell(a).then(|| a.value != Fp::zero())
    }

    /// The interval that narrows a memory read's choice set: the provenance
    /// table's interval of an IR value (read live, so later asserts are
    /// visible) intersected with the cell's proven range. Only for
    /// narrowing under a membership assertion, where a wrong interval makes
    /// the circuit unsatisfiable; comparisons use `proven_range` alone.
    fn interval_of(&self, cell: &Halo2BaseCell) -> Option<Interval> {
        let provenance = self
            .vid_of_cell
            .get(&cell.id)
            .and_then(|vid| self.prov.interval_of(*vid))
            .and_then(|iv| Interval::new(iv.lo as i128, iv.hi as i128));
        match (provenance, self.proven_range(cell)) {
            (Some(p), Some(r)) => Some(p.intersect(&r).unwrap_or(r)),
            (p, r) => p.or(r),
        }
    }

    fn wrap(&mut self, assigned: AssignedValue<Fp>, value: Fp) -> Halo2BaseCell {
        let id = self.next_id;
        self.next_id += 1;
        Halo2BaseCell { id, assigned: Some(assigned), value }
    }

    fn mark_bit(&mut self, cell: &Halo2BaseCell) {
        self.bit_cell_ids.insert(cell.id);
        if let Some(ptr) = cell.assigned.and_then(|a| a.cell) {
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
            .and_then(|a| a.cell)
            .is_some_and(|ptr| self.bit_cells.contains(&ptr))
    }

    fn is_const_cell(&self, cell: &Halo2BaseCell) -> bool {
        self.const_cell_ids.contains(&cell.id)
    }

    /// `cell` as a comparison operand: its value when it is a constant in
    /// the signed 64-bit window, the cell otherwise.
    fn operand(&self, cell: &Halo2BaseCell) -> Operand {
        match kernel::fp_to_i128(cell.value) {
            Some(v) if self.is_const_cell(cell) && Interval::point(v).is_some_and(|p| p.within(&Interval::SIGNED64)) => {
                Operand::Constant(v)
            }
            _ => Operand::Cell(*cell),
        }
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

    /// `first + b·c` by halo2-base's vertical gate `[first, b, c, out]`.
    /// When `first` is the last advice cell of the context (it was the
    /// output of the previous `gate_from`), the gate starts at that cell
    /// instead of copying it: three cells instead of four. halo2-base lays
    /// out gates that share a cell (its inner products do the same) and
    /// copies a shared cell across a column break. The decision depends
    /// only on the order of emission, never on values, so key generation
    /// and proving lay out the same gates.
    fn gate_from(&mut self, first: &Halo2BaseCell, b: QuantumCell<Fp>, c: QuantumCell<Fp>, value: Fp) -> Halo2BaseCell {
        let ctx = self.builder.main(0);
        let assigned = if self.last_output == Some((first.id, ctx.advice.len())) {
            ctx.assign_region_last([b, c, QuantumCell::Witness(value)], [-1])
        } else {
            ctx.assign_region_last([first.quantum(), b, c, QuantumCell::Witness(value)], [0])
        };
        let out = self.wrap(assigned, value);
        self.last_output = Some((out.id, self.builder.main(0).advice.len()));
        out
    }

    /// A value with no assigned cell, for a statement the plan proves is
    /// only read through facts (two-valued, range); like a deferred product,
    /// a gate that reached it would stop key generation.
    fn cellless(&mut self, value: Fp) -> Halo2BaseCell {
        let placeholder = AssignedValue { value: halo2_base::halo2_proofs::plonk::Assigned::Trivial(value), cell: None };
        self.wrap(placeholder, value)
    }

    /// A stand-in for the product `a·b` of a `MulI` that the plan fuses into
    /// its only consumer (an `AddI`, or a `SubI` subtracting it): no cell is
    /// assigned (`cell: None`); the consumer emits `acc ± a·b` as one gate
    /// from the recorded factors. The plan admits only products that no
    /// other statement and no affine view of a memory address reads, so the
    /// stand-in never reaches a gate; if it did, key generation would stop
    /// on the missing cell instead of building a circuit.
    fn defer_product(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Halo2BaseCell {
        let value = a.value * b.value;
        let placeholder = AssignedValue { value: halo2_base::halo2_proofs::plonk::Assigned::Trivial(value), cell: None };
        let cell = self.wrap(placeholder, value);
        self.deferred.insert(cell.id, (*a, *b));
        cell
    }

    /// A stand-in for an address value of `Plan::address_only`: no cell is
    /// assigned. Memory accesses read it through its affine view; one that
    /// needs the address as a cell calls [`Self::materialize_address`]. The
    /// plan admits only values whose every use is an address or another such
    /// value, so no gate reads the stand-in; if one did, key generation would
    /// stop on the missing cell instead of building a circuit.
    fn address_placeholder(&mut self, value: Fp, constant: bool) -> Halo2BaseCell {
        let placeholder = AssignedValue { value: halo2_base::halo2_proofs::plonk::Assigned::Trivial(value), cell: None };
        let cell = self.wrap(placeholder, value);
        if constant {
            self.const_cell_ids.insert(cell.id);
        }
        self.address_placeholders.insert(cell.id, None);
        cell
    }

    pub(super) fn is_address_placeholder(&self, cell: &Halo2BaseCell) -> bool {
        self.address_placeholders.contains_key(&cell.id)
    }

    /// `cell`, or for an address placeholder the address as a cell: the
    /// shared constant, or `Σ root·scale + offset` from its affine view
    /// (computed once per placeholder). The view is the provenance table's
    /// record of how the IR computed the value, so the cell equals it.
    pub(super) fn materialize_address(&mut self, cell: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        match self.address_placeholders.get(&cell.id) {
            None => return Ok(*cell),
            Some(Some(done)) => return Ok(*done),
            Some(None) => {}
        }
        let out = if self.is_const_cell(cell) {
            let c = self.const_cell(cell.value);
            c
        } else {
            let view = self
                .vid_of_cell
                .get(&cell.id)
                .and_then(|vid| self.prov.affine_of(*vid))
                .cloned()
                .ok_or_else(|| ProvingError::synthesis("address placeholder without an affine view"))?;
            let mut acc: Option<Halo2BaseCell> = None;
            for (root, scale) in &view.terms {
                let root = *self.cell_of_vid.get(root).ok_or_else(|| ProvingError::synthesis("address root without a cell"))?;
                let scale_fp = kernel::i64_to_fp(*scale);
                let (base_q, base_v) = match &acc {
                    Some(a) => (a.quantum(), a.value),
                    None => (QuantumCell::Constant(kernel::i64_to_fp(view.offset)), kernel::i64_to_fp(view.offset)),
                };
                let value = base_v + root.value * scale_fp;
                let next = self.gate_cell(value, |gate, ctx| gate.mul_add(ctx, root.quantum(), QuantumCell::Constant(scale_fp), base_q));
                acc = Some(next);
            }
            let out = acc.ok_or_else(|| ProvingError::synthesis("address placeholder without roots"))?;
            if let Some(r) = self.proven.get(&cell.id).copied() {
                self.proven.insert(out.id, r);
            }
            out
        };
        self.address_placeholders.insert(cell.id, Some(out));
        Ok(out)
    }

    /// The last select of a one-hot chain (plan.rs `OneHotChain`): the inner
    /// product of the equality bits with the link values, the missing value's
    /// bit standing for the innermost default. Sound because the planned
    /// bound `AssertU128Lt(root, n)` (emitted wherever it appears) leaves the
    /// root one of `0..n`, so exactly one bit is 1 and the product equals the
    /// select chain's value. The range is the hull of the values' ranges.
    fn onehot_chain(
        &mut self,
        chain: &plan::OneHotChain,
        last_cond: &Halo2BaseCell,
        last_value: &Halo2BaseCell,
    ) -> Result<Halo2BaseCell, ProvingError> {
        let cell = |s: &Self, vid: &crate::types::ValueId| {
            s.cell_of_vid.get(vid).copied().ok_or_else(|| ProvingError::synthesis("one-hot chain operand without a cell"))
        };
        let root = cell(self, &chain.root)?;
        let default = cell(self, &chain.default)?;
        let mut choices = vec![(self.root_indicator(&root, chain.missing)?, default)];
        for (cond, value) in &chain.terms[..chain.terms.len() - 1] {
            choices.push((cell(self, cond)?, cell(self, value)?));
        }
        choices.push((*last_cond, *last_value));
        let ranges: Option<Vec<Interval>> = choices.iter().map(|(_, v)| self.proven_range(v)).collect();
        let out = self.one_hot_combination(&choices);
        if let Some(hull) = ranges.and_then(|rs| rs.into_iter().reduce(|a, b| a.hull(&b))) {
            self.prove_range(&out, hull);
        }
        Ok(out)
    }

    /// `cell` itself, or for a deferred product its product as a cell (when
    /// both operands of a sum are deferred products, one is emitted).
    fn materialize(&mut self, cell: &Halo2BaseCell) -> Halo2BaseCell {
        let Some((a, b)) = self.deferred.get(&cell.id).copied() else { return *cell };
        // `0 + a·b` as the first gate of a chain: the multiply-add that
        // consumes it starts at its output cell (three cells, not four).
        self.product(&a, &b)
    }

    /// `a·b` as the first gate of a multiply-add chain (`0 + a·b`, so the
    /// next `multiply_add` can start at its output cell).
    fn product(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Halo2BaseCell {
        let zero = self.const_cell(Fp::zero());
        let out = self.gate_from(&zero, a.quantum(), b.quantum(), a.value * b.value);
        self.propagate_range(&out, a, b, Interval::mul);
        out
    }

    /// `acc + a·b` in one gate, with the range of the sum propagated.
    fn multiply_add(&mut self, acc: &Halo2BaseCell, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Halo2BaseCell {
        let out = self.gate_from(acc, a.quantum(), b.quantum(), acc.value + a.value * b.value);
        if let (Some(ra), Some(rb), Some(racc)) = (self.proven_range(a), self.proven_range(b), self.proven_range(acc)) {
            if let Some(range) = ra.mul(&rb).and_then(|p| p.add(&racc)) {
                self.prove_range(&out, range);
            }
        }
        out
    }

    fn truthy_bit(&mut self, a: &Halo2BaseCell) -> Halo2BaseCell {
        if self.is_bit(a) {
            return *a;
        }
        // `bit ? v1 : v0` is non-zero exactly when the bit selects a
        // non-zero value (the two values differ).
        if let Some(tv) = self.two_valued_of(a) {
            match (tv.if_zero != 0, tv.if_one != 0) {
                (false, true) => return tv.bit,
                (true, false) => {
                    let out = self.gate_cell(Fp::one() - tv.bit.value, |gate, ctx| gate.not(ctx, tv.bit.quantum()));
                    self.mark_bit(&out);
                    return out;
                }
                _ => {}
            }
        }
        self.is_nonzero_bit(a)
    }

    fn constrain_equal(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) {
        let (a, b) = (self.assigned(a), self.assigned(b));
        self.ctx().constrain_equal(&a, &b);
    }

    /// Constrain `a == c`. A constant `a` equal to `c` holds by construction
    /// and needs no constraint (the comparisons an `Assert` consumes return
    /// such a constant when their own checks already prove the relation, see
    /// `ranges::assert_relation`); any other constant is still constrained,
    /// so a false assertion leaves the circuit unsatisfiable.
    fn assert_const(&mut self, a: &Halo2BaseCell, c: Fp) {
        if self.is_const_cell(a) && a.value == c {
            return;
        }
        let a = self.assigned(a);
        self.gate.assert_is_const(self.builder.main(0), &a, &c);
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

    /// `a != b` as an assertion, for a `!=` whose only use is `Assert`: the
    /// hint `inv` with `(a - b)·inv = 1`, which has a solution exactly when
    /// `a - b ≠ 0` (and then only `inv = 1/(a - b)`). Records `a` (or `b`)
    /// as non-zero when the other is the constant 0, and with it the integer
    /// a float cast stands for. Returns the constant `true`.
    fn assert_distinct(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let zero = |s: &Self, x: &Halo2BaseCell| s.is_const_cell(x) && x.value == Fp::zero();
        // `d != 0` before a division by `d` (`Plan::divisor_guards`): record
        // `d` as non-zero and emit nothing. The division then takes
        // `div_rem_nonzero`, whose remainder bound `0 <= |r| <= |d| - 1`
        // (`assert_below_cell`) has no solution for `d = 0`; a circuit has no
        // control flow, so that constraint holds wherever it is emitted.
        if self.plan.divisor_guards.contains(&self.current_stmt) {
            let divisor = if zero(self, b) { Some(*a) } else if zero(self, a) { Some(*b) } else { None };
            if let Some(d) = divisor.filter(|d| !self.is_const_cell(d)) {
                if d.value == Fp::zero() {
                    self.note_unsat(|| "Witness does not satisfy circuit constraints: an asserted `!=` is false".to_string());
                }
                self.record_nonzero(&d);
                return self.constant_bool(true);
            }
        }
        let difference = if zero(self, b) {
            *a
        } else if zero(self, a) {
            *b
        } else {
            self.difference(a, b)
        };
        if difference.value == Fp::zero() {
            self.note_unsat(|| "Witness does not satisfy circuit constraints: an asserted `!=` is false".to_string());
        }
        let inverse = self.hint(difference.value.invert().unwrap_or(Fp::zero()));
        let product = self.gate_cell(difference.value * inverse.value, |gate, ctx| gate.mul(ctx, difference.quantum(), inverse.quantum()));
        self.assert_const(&product, Fp::one());
        if zero(self, b) || zero(self, a) {
            self.record_nonzero(&difference);
        }
        self.constant_bool(true)
    }

    /// Record that the constraints (or a later division's) make `d` non-zero,
    /// and with it the integer `n` of a float cast `d = n·2^P`.
    fn record_nonzero(&mut self, d: &Halo2BaseCell) {
        self.nonzero.insert(d.id);
        if let Some(n) = self.float_casts.get(&d.id).copied() {
            self.nonzero.insert(n.id); // n·2^P ≠ 0 exactly when n ≠ 0
        }
    }

    /// `a == b` as an assertion, for an `==` whose only use is `Assert`: a
    /// copy constraint between the two cells, or of the cell to a fixed
    /// constant (no advice cell either way), which holds exactly when the
    /// field elements are equal, the meaning of `EqI`/`EqF`. Returns the
    /// constant `true`.
    fn assert_equal(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        if a.value != b.value {
            self.note_unsat(|| "Witness does not satisfy circuit constraints: an asserted `==` is false".to_string());
        }
        let (a, b) = (self.materialize(a), self.materialize(b));
        if self.is_const_cell(&b) {
            self.assert_const(&a, b.value);
        } else if self.is_const_cell(&a) {
            self.assert_const(&b, a.value);
        } else {
            self.constrain_equal(&a, &b);
        }
        self.constant_bool(true)
    }

    /// `a - b`, or the other operand when one is the constant 0 (an
    /// equality test with 0 tests the value itself).
    fn difference_or_operand(&mut self, a: &Halo2BaseCell, b: &Halo2BaseCell) -> Halo2BaseCell {
        let zero = |s: &Self, x: &Halo2BaseCell| s.is_const_cell(x) && x.value == Fp::zero();
        if zero(self, b) {
            *a
        } else if zero(self, a) {
            *b
        } else {
            self.difference(a, b)
        }
    }

    /// `[x != 0]` as a marked bit, in eight cells: `nz = x·inv` and
    /// `x·nz = x` (gates `[0, x, inv, nz]` and `[0, x, nz, x]`). For `x ≠ 0`
    /// the second gate forces `nz = 1`; for `x = 0` the first forces
    /// `nz = 0`. (`[x == 0]` and a negation cost eleven.) `inv` is the
    /// gadget's witness, free only when `x = 0`, where it does not matter.
    fn is_nonzero_bit(&mut self, x: &Halo2BaseCell) -> Halo2BaseCell {
        use halo2_base::halo2_proofs::plonk::Assigned;
        let (value, inverse) = if x.value == Fp::zero() {
            (Fp::zero(), Assigned::Trivial(Fp::zero()))
        } else {
            (Fp::one(), Assigned::Rational(Fp::one(), x.value))
        };
        let ctx = self.builder.main(0);
        let cells = [
            QuantumCell::Constant(Fp::zero()),
            x.quantum(),
            QuantumCell::WitnessFraction(inverse),
            QuantumCell::Witness(value),
            QuantumCell::Constant(Fp::zero()),
            x.quantum(),
            QuantumCell::Witness(value),
            x.quantum(),
        ];
        ctx.assign_region_smart(cells, [0, 4], [(3, 6)], []);
        let assigned = ctx.get(-5);
        let bit = self.wrap(assigned, value);
        self.mark_bit(&bit);
        bit
    }

    /// `[x == 0]` as a marked bit: halo2-base's inverse-witness gadget,
    /// `z + x·inv = 1` and `x·z = 0`, laid out in seven cells
    /// `[z, x, inv, 1, x, z, 1]` with gates at 0 and 3 and the two `z` cells
    /// equal (halo2-base's own layout spends an eighth cell on a zero).
    /// For `x ≠ 0` the second gate forces `z = 0`; for `x = 0` the first
    /// forces `z = 1`. `inv` is the gadget's witness (`1/x`, or 1 for 0); any
    /// other value leaves no satisfying `z` or the same one.
    fn is_zero_bit(&mut self, x: &Halo2BaseCell) -> Halo2BaseCell {
        use halo2_base::halo2_proofs::plonk::Assigned;
        let (value, inverse) = if x.value == Fp::zero() {
            (Fp::one(), Assigned::Trivial(Fp::one()))
        } else {
            (Fp::zero(), Assigned::Rational(Fp::one(), x.value))
        };
        let ctx = self.builder.main(0);
        let cells = [
            QuantumCell::Witness(value),
            x.quantum(),
            QuantumCell::WitnessFraction(inverse),
            QuantumCell::Constant(Fp::one()),
            x.quantum(),
            QuantumCell::Witness(value),
            QuantumCell::Constant(Fp::one()),
        ];
        ctx.assign_region_smart(cells, [0, 3], [(0, 5)], []);
        let assigned = ctx.get(-7);
        let bit = self.wrap(assigned, value);
        self.mark_bit(&bit);
        bit
    }

    fn negate(&mut self, value: &Halo2BaseCell) -> Halo2BaseCell {
        let out = self.gate_cell(-value.value, |gate, ctx| gate.neg(ctx, value.quantum()));
        if let Some(range) = self.proven_range(value) {
            self.prove_range(&out, range.neg());
        }
        out
    }

    /// `sign ? -value : value` for a bit `sign`.
    fn negate_if(&mut self, value: &Halo2BaseCell, sign: &Halo2BaseCell) -> Result<Halo2BaseCell, ProvingError> {
        let neg = self.negate(value);
        self.select_bit(sign, &neg, value)
    }

}

impl Synthesizer for Halo2BaseSynthesizer<'_> {
    fn on_stmt_begin(&mut self, stmt_id: u32) {
        self.current_stmt = stmt_id;
        if self.profile.is_none() {
            return;
        }
        self.close_profile_entry();
        let (advice, lookup) = self.cell_counters();
        if let Some(profile) = self.profile.as_mut() {
            profile.current = Some((stmt_id, advice, lookup));
        }
    }

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
        if self.plan.address_only.contains(&self.current_stmt) {
            return Ok(self.address_placeholder(kernel::i64_to_fp(value), true));
        }
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
        if self.plan.address_only.contains(&self.current_stmt) {
            let out = self.address_placeholder(a.value + b.value, false);
            self.propagate_range(&out, a, b, Interval::add);
            return Ok(out);
        }
        if let Some((x, y)) = self.deferred.get(&b.id).copied() {
            let acc = self.materialize(a);
            return Ok(self.multiply_add(&acc, &x, &y));
        }
        if let Some((x, y)) = self.deferred.get(&a.id).copied() {
            return Ok(self.multiply_add(b, &x, &y));
        }
        let out = self.gate_from(a, b.quantum(), QuantumCell::Constant(Fp::one()), a.value + b.value);
        // out - a = b and out - b = a: a select between the sum and one of
        // its summands needs no difference gate (see `select_bit`).
        self.differences.entry((out.id, a.id)).or_insert(*b);
        self.differences.entry((out.id, b.id)).or_insert(*a);
        self.propagate_range(&out, a, b, Interval::add);
        self.propagate_two_valued(&out, a, b, |x, c| x.checked_add(c));
        self.propagate_two_valued(&out, b, a, |x, c| x.checked_add(c));
        Ok(out)
    }

    fn sub_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.plan.address_only.contains(&self.current_stmt) {
            let out = self.address_placeholder(a.value - b.value, false);
            self.propagate_range(&out, a, b, Interval::sub);
            return Ok(out);
        }
        if let Some((x, y)) = self.deferred.get(&b.id).copied() {
            // `a - x·y` as halo2-base's `[a - x·y, x, y, a]` gate.
            let a = &self.materialize(a);
            let out = self.gate_cell(a.value - x.value * y.value, |gate, ctx| gate.sub_mul(ctx, a.quantum(), x.quantum(), y.quantum()));
            if let (Some(ra), Some(rx), Some(ry)) = (self.proven_range(a), self.proven_range(&x), self.proven_range(&y)) {
                if let Some(range) = rx.mul(&ry).and_then(|p| ra.sub(&p)) {
                    self.prove_range(&out, range);
                }
            }
            return Ok(out);
        }
        // A fresh cell per IR value (provenance maps cells to values one to
        // one); later selects on the same operands may reuse it.
        let out = self.gate_cell(a.value - b.value, |gate, ctx| gate.sub(ctx, a.quantum(), b.quantum()));
        self.differences.entry((a.id, b.id)).or_insert(out);
        self.differences.entry((a.id, out.id)).or_insert(*b); // a - out = b
        self.propagate_range(&out, a, b, Interval::sub);
        self.propagate_two_valued(&out, a, b, |x, c| x.checked_sub(c));
        self.propagate_two_valued(&out, b, a, |x, c| c.checked_sub(x));
        Ok(out)
    }

    fn mul_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.plan.address_only.contains(&self.current_stmt) {
            let out = self.address_placeholder(a.value * b.value, false);
            self.propagate_range(&out, a, b, Interval::mul);
            return Ok(out);
        }
        if self.plan.deferred_products.contains(&self.current_stmt) {
            return Ok(self.defer_product(a, b));
        }
        let out = self.gate_cell(a.value * b.value, |gate, ctx| gate.mul(ctx, a.quantum(), b.quantum()));
        self.propagate_range(&out, a, b, Interval::mul);
        self.propagate_two_valued(&out, a, b, |x, c| x.checked_mul(c));
        self.propagate_two_valued(&out, b, a, |x, c| x.checked_mul(c));
        Ok(out)
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
            gate.mul_not(ctx, b_is_zero.quantum(), a.quantum())
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
        self.ensure_int64_divisor(b)?;
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
            gate.mul_not(ctx, b_is_zero.quantum(), a.quantum())
        });
        let q_value = match (kernel::fp_to_i128(a_masked.value), kernel::fp_to_i128(b_nonzero.value)) {
            (Some(av), Some(bv)) => kernel::div_rem_or_zero(av, bv, Rounding::Floor).0,
            _ => 0,
        };
        let q = self.hint(kernel::i128_to_fp(q_value));
        let r = self.gate_cell(a_masked.value - b_nonzero.value * q.value, |gate, ctx| {
            gate.sub_mul(ctx, a_masked.quantum(), b_nonzero.quantum(), q.quantum())
        });
        self.range_check(&q, bits);
        self.range_check(&r, bits);
        let gap = self.gate_cell(b_nonzero.value - Fp::one() - r.value, |gate, ctx| {
            let b_minus_r = gate.sub(ctx, b_nonzero.quantum(), r.quantum());
            gate.dec(ctx, b_minus_r)
        });
        self.range_check(&gap, bits);
        Ok(q)
    }

    /// Python remainder `a % b` (sign of the divisor).
    fn mod_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.ensure_int64_divisor(b)?;
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
        let exp_cell = self.assigned(exp);
        let bits = self.gate.num_to_bits(self.builder.main(0), exp_cell, DYNAMIC_EXP_BITS);
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
        self.magnitude(a)
    }

    /// `sign(a) = [a != 0] - 2 * [a < 0]`, from the pinned sign bit of
    /// `sign_and_abs`.
    fn sign_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let (negative, _) = self.sign_and_abs(a)?;
        let is_zero = self.is_zero_bit(a);
        let nonzero = self.logical_not(&is_zero)?;
        let two = Fp::from(2u64);
        Ok(self.gate_cell(nonzero.value - two * negative.value, |gate, ctx| {
            gate.sub_mul(ctx, nonzero.quantum(), negative.quantum(), QuantumCell::Constant(two))
        }))
    }

    /// `1 / a` in the field (the IR's `InvI`); 0 for `a = 0`, as in the mock.
    fn inv_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let one = self.constant_int(1)?;
        self.div_i(&one, a)
    }

    fn bit_and_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.bit_and(a, b) }
    fn bit_or_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.bit_or(a, b) }
    fn bit_xor_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.bit_xor(a, b) }
    fn shl_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.shift_left(a, b) }
    fn shr_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.shift_right(a, b) }
    fn bit_not_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.bit_not(a) }

    fn add_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.add_i(a, b) }
    fn sub_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.sub_i(a, b) }

    fn mul_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.fixed_mul(a, b)
    }

    fn product_sum_f(&mut self, pairs: &[(Self::CellRef, Self::CellRef)]) -> Result<Self::CellRef, ProvingError> {
        self.fixed_product_sum(pairs)
    }

    fn div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.fixed_div(a, b)
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
        self.fixed_pow(base, exp)
    }
    fn pow_f_int(&mut self, base: &Self::CellRef, exp: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.fixed_pow_int(base, exp)
    }
    fn abs_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.abs_i(a) }
    fn sign_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.sign_i(a) }

    fn eq_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            return self.assert_equal(a, b);
        }
        if let Some(bit) = self.equals_constant(a, b)? {
            return Ok(bit);
        }
        if let Some(bit) = self.equals_constant(b, a)? {
            return Ok(bit);
        }
        // `[a == b] = [a - b == 0]`.
        let difference = self.difference_or_operand(a, b);
        Ok(self.is_zero_bit(&difference))
    }
    fn ne_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            return self.assert_distinct(a, b);
        }
        // A two-valued cell against a constant: the negated bit (or a constant).
        for (x, c) in [(a, b), (b, a)] {
            if let Some(eq) = self.equals_constant(x, c)? {
                return self.logical_not(&eq);
            }
        }
        let difference = self.difference_or_operand(a, b);
        Ok(self.is_nonzero_bit(&difference))
    }
    /// Signed `a < b`; operands outside the signed 64-bit window have no
    /// witness (see `ranges::less_than`).
    fn lt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            return self.assert_relation(a, b, true);
        }
        self.less_than(self.operand(a), self.operand(b), Some(Interval::SIGNED64))
    }
    /// Unsigned `a < b` for operands in `[0, 2^64)` (each is range-checked
    /// there unless already known to lie inside).
    fn lt_i_bounded(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.less_than(self.operand(a), self.operand(b), Some(Interval::UNSIGNED64))
    }
    fn gt_i_bounded(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i_bounded(b, a) }
    fn lte_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            return self.assert_relation(a, b, false);
        }
        let gt = self.gt_i(a, b)?;
        self.logical_not(&gt)
    }
    fn gt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i(b, a) }
    fn gte_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            if let Some(holds) = self.assert_nonneg_by_root(a, b)? {
                return Ok(holds);
            }
            return self.assert_relation(b, a, false);
        }
        let lt = self.lt_i(a, b)?;
        self.logical_not(&lt)
    }
    fn eq_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.eq_i(a, b) }
    fn ne_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.ne_i(a, b) }
    fn lt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i(a, b) }
    fn lte_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            if let Some(holds) = self.assert_nonneg_by_root(b, a)? {
                return Ok(holds);
            }
            return self.assert_relation(a, b, false);
        }
        let gt = self.gt_f(a, b)?;
        self.logical_not(&gt)
    }
    fn gt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.lt_i(b, a) }
    fn gte_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.is_asserted_comparison() {
            if let Some(holds) = self.assert_nonneg_by_root(a, b)? {
                return Ok(holds);
            }
            return self.assert_relation(b, a, false);
        }
        let lt = self.lt_f(a, b)?;
        self.logical_not(&lt)
    }

    /// `sin` by the algorithm of `kernel::fp_sin`: range reduction modulo
    /// `2 * pi_q`, fold onto `[0, π)`, mirror onto `[0, π/2]`, then
    /// `s * P(s^2)` with the least-squares polynomial `SIN_ODD_U_COEFS`.
    ///
    /// Soundness: `sign_and_abs`, the division by the constant `2 * pi_q`
    /// and every `mul_f` are functions of their inputs (see each). The two
    /// comparisons take operands already proven to lie in `[0, 2 * pi_q)`.
    /// The rest are gates and selects on pinned bits, so the output is a
    /// function of `a`.
    fn sin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let pi = kernel::pi_q(precision_bits);
        let half_pi = kernel::half_pi_q(precision_bits);
        let (negative, magnitude) = self.sign_and_abs(a)?;
        let (_, r) = self.div_rem_nonzero(&magnitude, Divisor::Constant(2 * pi), Rounding::Euclid)?;

        // r in [0, 2π): fold onto [0, π).
        let below_pi = self.less_than(Operand::Cell(r), Operand::Constant(pi), None)?;
        let folded_bit = self.logical_not(&below_pi)?;
        let pi_fp = kernel::i128_to_fp(pi);
        let r = self.gate_cell(r.value - folded_bit.value * pi_fp, |gate, ctx| {
            gate.sub_mul(ctx, r.quantum(), folded_bit.quantum(), QuantumCell::Constant(pi_fp))
        });
        // r - [r >= π]·π for r in [0, 2π) lies in [0, π).
        self.prove_range(&r, Interval { lo: 0, hi: pi - 1 });
        // [0, π): mirror onto [0, π/2].
        let up_to_half_pi = self.less_than(Operand::Cell(r), Operand::Constant(half_pi + 1), None)?;
        let mirrored = self.constant_minus(pi, &r);
        let s = self.select_bit(&up_to_half_pi, &r, &mirrored)?;
        // r if r <= π/2, else π - r < π/2 (r > π/2): in [0, π/2].
        self.prove_range(&s, Interval { lo: 0, hi: half_pi });

        let u = self.mul_f(&s, &s)?;
        let odd = self.poly_eval_horner(&u, &kernel::sin_odd_coefs_q(precision_bits))?;
        let value = self.mul_f(&s, &odd)?;

        let flip_value = negative.value + folded_bit.value - Fp::from(2u64) * negative.value * folded_bit.value;
        let flip = self.gate_cell(flip_value, |gate, ctx| gate.xor(ctx, negative.quantum(), folded_bit.quantum()));
        self.mark_bit(&flip);
        self.negate_if(&value, &flip)
    }
    /// `sinh x = floor((exp x - exp -x) / 2)`, as `kernel::fp_sinh`: two
    /// pinned `exp`s and a pinned rescale by one bit.
    fn sinh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.half_exp_combination(a, false) }
    /// `cos(a) = sin(a + pi/2)`, as `kernel::fp_cos`.
    fn cos_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let shifted = self.add_constant(a, kernel::half_pi_q(self.params.precision_bits));
        self.sin_f(&shifted)
    }
    /// `cosh x = floor((exp x + exp -x) / 2)`, as `kernel::fp_cosh`.
    fn cosh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.half_exp_combination(a, true) }
    fn tan_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("tan_f") }
    /// `tanh x = sinh x / cosh x` (the fixed-point quotient), as
    /// `kernel::fp_tanh`; the division pins its quotient.
    fn tanh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let sinh = self.sinh_f(a)?;
        let cosh = self.cosh_f(a)?;
        self.fixed_div(&sinh, &cosh)
    }
    /// `sqrt` as `kernel::fp_sqrt`: `floor(sqrt(a * 2^P))`, and 0 for a
    /// negative input. The pinned sign bit of `sign_and_abs` zeroes a
    /// negative input; `sqrt_nonneg` pins the root.
    /// An `a` proven non-negative (for example by the domain assertion the
    /// frontend emits before `np.sqrt`) needs no sign handling, and its
    /// range bounds the root's checks.
    fn sqrt_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if let Some(range) = self.proven_range(a).filter(|r| r.lo >= 0 && r.hi < 1i128 << 64) {
            let bits = (128 - range.hi.leading_zeros()) as usize;
            return self.sqrt_nonneg(a, bits);
        }
        let (negative, _) = self.sign_and_abs(a)?;
        let clamped = self.gate_cell((Fp::one() - negative.value) * a.value, |gate, ctx| {
            gate.mul_not(ctx, negative.quantum(), a.quantum())
        });
        self.sqrt_nonneg(&clamped, 64)
    }
    fn sqrt_product_i_f(&mut self, b: &Self::CellRef, c: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.sqrt_product_int(b, c)
    }
    /// `exp`, by `kernel::fp_exp`'s algorithm (Noir's `qexp` is the same):
    /// `y = mul_f(x, 1/ln 2)`, `(i, f) = (floor(y / 2^P), y mod 2^P)`,
    /// `t = 2^(f / 2^P)` by Horner, and with `k = i + EXP_SHIFT` the result
    /// is 0 for `k < 0` and `floor(t·2^k / 2^EXP_SHIFT)` otherwise.
    ///
    /// Soundness: `mul_f`, the two rescales and every Horner step pin their
    /// quotients, so `y`, `i`, `f` and `t` are functions of `x`. The
    /// underflow bit `u = [k < 0]` is a pinned comparison. `k' = (1 - u)·k`
    /// is decomposed into 7 bits (`power_of_two`), which proves
    /// `0 <= k' < 128`: there is no witness for `k >= 128`, where the result
    /// leaves the window anyway. `2^k' < 2^128` and `t < 2^(P+2)`, so the gate
    /// product is the integer `t·2^k'` (below p), and `rescale_by` pins its
    /// floor and checks it into the signed 64-bit window: no witness when the
    /// result leaves it (the kernel's `None`). The output `(1 - u)·q` is a
    /// function of `x`: the kernel's value.
    fn exp_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        if kernel::fp_exp(a.value, precision_bits).is_none() {
            let x = kernel::dequantize_from_fp(a.value, precision_bits);
            self.note_unsat(move || format!("exp({x}) is outside the fixed-point domain (result beyond the signed 64-bit window)"));
        }
        let inv_ln2 = self.const_cell(kernel::i128_to_fp(kernel::inv_ln2_q(precision_bits)));
        let y = self.mul_f(a, &inv_ln2)?;
        let (i, f) = self.rescale_with_remainder(&y, precision_bits)?;
        let t = self.poly_eval_horner(&f, &kernel::exp2_coefs_q(precision_bits))?;
        let k = self.add_constant(&i, kernel::EXP_SHIFT);
        let underflow = self.less_than(Operand::Cell(k), Operand::Constant(0), Some(Interval::SIGNED64))?;
        let kept = self.gate_cell((Fp::one() - underflow.value) * k.value, |gate, ctx| {
            gate.mul_not(ctx, underflow.quantum(), k.quantum())
        });
        let power = self.power_of_two(&kept, 7);
        let raw = self.gate_cell(t.value * power.value, |gate, ctx| gate.mul(ctx, t.quantum(), power.quantum()));
        let q = self.rescale_by(&raw, kernel::EXP_SHIFT as u32)?;
        let out = self.gate_cell((Fp::one() - underflow.value) * q.value, |gate, ctx| {
            gate.mul_not(ctx, underflow.quantum(), q.quantum())
        });
        self.prove_range(&out, Interval { lo: 0, hi: (1i128 << 63) - 1 });
        Ok(out)
    }
    /// `log`, as `kernel::fp_log` (see `fixed_log`).
    fn log_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.fixed_log(a) }
    /// `arcsin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))`, as `kernel::fp_arcsin`
    /// (0 outside `[-1, 1]`).
    ///
    /// Soundness: every step is a function of its inputs. The domain bit
    /// compares the pinned magnitude `|x| <= 2^64` against `1` within 65
    /// bits. With `x'` = x inside the domain and 0 outside, `2^(2P) - x'^2`
    /// (one gate) lies in `[0, 2^(2P)]`, meeting `isqrt`'s precondition; the
    /// divisor `2^P + root` is positive, and the remainder bound of
    /// `div_rem_nonzero` re-proves it.
    fn arcsin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let precision_bits = self.params.precision_bits;
        let one = kernel::scale_fp(precision_bits);
        let magnitude = self.magnitude(a)?;
        let in_domain = self.less_than(Operand::Cell(magnitude), Operand::Constant((1i128 << precision_bits) + 1), None)?;
        let x = self.mul_i(&in_domain, a)?;

        let one_squared = one * one;
        let radicand = self.gate_cell(one_squared - x.value * x.value, |gate, ctx| {
            gate.sub_mul(ctx, QuantumCell::Constant(one_squared), x.quantum(), x.quantum())
        });
        let root = self.isqrt(&radicand, 2 * precision_bits as usize + 1)?;
        let denominator = self.gate_cell(root.value + one, |gate, ctx| {
            gate.add(ctx, root.quantum(), QuantumCell::Constant(one))
        });
        let scaled = self.gate_cell(x.value * one, |gate, ctx| {
            gate.mul(ctx, x.quantum(), QuantumCell::Constant(one))
        });
        let (t, _) = self.div_rem_nonzero(&scaled, Divisor::Positive(denominator), Rounding::Euclid)?;

        let t_squared = self.mul_f(&t, &t)?;
        let odd_part = self.poly_eval_horner(&t_squared, &kernel::atan_odd_coefs_q(precision_bits))?;
        let half = self.mul_f(&t, &odd_part)?;
        self.add_i(&half, &half)
    }
    fn arccos_f(&mut self, _: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.unsupported("arccos_f") }
    fn arctan2_f(&mut self, y: &Self::CellRef, x: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { self.fixed_atan2(y, x) }

    fn logical_and(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        // `x and true = [x]`, `x and false = false`.
        for (constant, other) in [(a, b), (b, a)] {
            if let Some(truth) = self.constant_truth(constant) {
                return if truth { Ok(self.truthy_bit(other)) } else { self.constant_bool(false) };
            }
        }
        let ta = self.truthy_bit(a);
        let tb = self.truthy_bit(b);
        let out = self.gate_cell(ta.value * tb.value, |gate, ctx| gate.and(ctx, ta.quantum(), tb.quantum()));
        self.mark_bit(&out);
        Ok(out)
    }
    fn logical_or(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        // `x or false = [x]`, `x or true = true`.
        for (constant, other) in [(a, b), (b, a)] {
            if let Some(truth) = self.constant_truth(constant) {
                return if truth { self.constant_bool(true) } else { Ok(self.truthy_bit(other)) };
            }
        }
        let ta = self.truthy_bit(a);
        let tb = self.truthy_bit(b);
        let out = self.gate_cell(ta.value + tb.value - ta.value * tb.value, |gate, ctx| gate.or(ctx, ta.quantum(), tb.quantum()));
        self.mark_bit(&out);
        Ok(out)
    }
    fn logical_not(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if let Some(truth) = self.constant_truth(a) {
            return self.constant_bool(!truth);
        }
        let ta = self.truthy_bit(a);
        let out = self.gate_cell(Fp::one() - ta.value, |gate, ctx| gate.not(ctx, ta.quantum()));
        self.mark_bit(&out);
        self.negations.insert(out.id, ta.id);
        Ok(out)
    }

    fn select(&mut self, cond: &Self::CellRef, if_true: &Self::CellRef, if_false: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        if self.plan.onehot_links.contains(&self.current_stmt) {
            // A link of a one-hot chain (plan.rs): only the chain's last select
            // reads it, through the chain's terms, so it needs no cell.
            let value = cond.value * if_true.value + (Fp::one() - cond.value) * if_false.value;
            let placeholder = AssignedValue { value: halo2_base::halo2_proofs::plonk::Assigned::Trivial(value), cell: None };
            return Ok(self.wrap(placeholder, value));
        }
        if let Some(chain) = self.plan.onehot_chains.get(&self.current_stmt).cloned() {
            return self.onehot_chain(&chain, cond, if_true);
        }
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
        // Otherwise `f + c·(t - f)`, one gate when the difference exists
        // already (the comparison that produced `c`, or the select with the
        // branches swapped, made it): `f - c·(f - t)` for the swapped one.
        let out = if t_zero {
            self.gate_cell(value, |gate, ctx| gate.mul_not(ctx, cond.quantum(), if_false.quantum()))
        } else if f_zero {
            self.gate_cell(value, |gate, ctx| gate.mul(ctx, cond.quantum(), if_true.quantum()))
        } else if let Some(swapped) = self.differences.get(&(if_false.id, if_true.id)).copied() {
            self.gate_cell(value, |gate, ctx| gate.sub_mul(ctx, if_false.quantum(), cond.quantum(), swapped.quantum()))
        } else {
            let diff = self.difference(if_true, if_false);
            self.gate_cell(value, |gate, ctx| gate.mul_add(ctx, cond.quantum(), diff.quantum(), if_false.quantum()))
        };
        // The select gate pins out to one of the branches (cond is a bit), so
        // the hull of the branches' proven ranges holds for the output, and
        // a select between bits is a bit.
        self.propagate_range(&out, if_true, if_false, |t, f| Some(t.hull(f)));
        if self.is_bit(if_true) && self.is_bit(if_false) {
            self.mark_bit(&out);
        }
        let constant = |s: &Self, x: &Halo2BaseCell| s.is_const_cell(x).then(|| kernel::fp_to_i128(x.value)).flatten();
        if let (Some(t), Some(f)) = (constant(self, if_true), constant(self, if_false)) {
            self.set_two_valued(&out, TwoValued { bit: *cond, if_zero: f, if_one: t });
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
        if let Some(n) = self.float_casts.get(&a.id) {
            return Ok(*n); // trunc(n·2^P / 2^P) = n
        }
        let scale = 1i128 << self.params.precision_bits;
        Ok(self.div_rem_nonzero(a, Divisor::Constant(scale), Rounding::Trunc)?.0)
    }
    /// Integer to Q value: `a · 2^P` (one gate). The result is remembered as
    /// a float cast of `a`, so products and quotients with it stay exact
    /// (see `fixed_point::Factor`).
    fn float_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let scale = self.constant_int(crate::prove::field::quantization_scale(self.params.precision_bits) as i64)?;
        let out = self.mul_i(a, &scale)?;
        self.float_casts.insert(out.id, *a);
        Ok(out)
    }
    fn bool_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError> { Ok(self.truthy_bit(a)) }

    fn read_input(&mut self, path: &InputPath, is_public: bool) -> Result<Self::CellRef, ProvingError> {
        let val = self.get_witness_fp_path(path).unwrap_or(Fp::zero());
        if !is_public {
            if let Some(&bound) = self.plan.bounded_inputs.get(&self.current_stmt) {
                if let Some(cell) = self.read_bounded_input(val, bound) {
                    return Ok(cell);
                }
            }
        }
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
        let assigned = self.assigned(a);
        self.builder.assigned_instances[0].push(assigned);
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
        self.refine_by_assertion(a);
        Ok(())
    }
    fn assert_u128_lt(&mut self, idx: &Self::CellRef, hi: &Self::CellRef) -> Result<(), ProvingError> {
        let (i, h) = (kernel::fp_to_i128(idx.value), kernel::fp_to_i128(hi.value));
        if !i.zip(h).is_some_and(|(i, h)| 0 <= i && i < h) {
            let (i, h) = (kernel::fp_to_i64(idx.value), kernel::fp_to_i64(hi.value));
            self.note_unsat(move || {
                format!("Witness does not satisfy circuit constraints: index {} is not below {}", i, h)
            });
        }
        self.assert_less(idx, self.operand(hi))
    }
    fn assert_u128_lt_bit(&mut self, idx: &Self::CellRef, hi: u64, bit: u32, bits: u32) -> Result<Self::CellRef, ProvingError> {
        let key = (idx.id, hi, bits);
        if !self.u128_lt_bit_cache.contains_key(&key) {
            if !kernel::fp_to_i128(idx.value).is_some_and(|i| 0 <= i && i < hi as i128) {
                let i = kernel::fp_to_i64(idx.value);
                self.note_unsat(move || format!("Witness does not satisfy circuit constraints: index {i} is not below {hi}"));
            }
            self.assert_less(idx, Operand::Constant(hi as i128))?;
            let idx_cell = self.assigned(idx);
            let decomposed = self.gate.num_to_bits(self.builder.main(0), idx_cell, bits as usize);
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
        self.range_check(a, 63);
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
        let a_cell = self.assigned(a);
        self.gate.assert_bit(self.builder.main(0), a_cell);
        self.mark_bit(a);
        Ok(())
    }

    fn allocate_memory(&mut self, segment_id: u32, size: u32, init: i64) -> Result<(), ProvingError> {
        self.memory_allocate(segment_id, size, init);
        Ok(())
    }
    fn write_memory(&mut self, segment_id: u32, addr: &Self::CellRef, value: &Self::CellRef) -> Result<(), ProvingError> {
        self.memory_write(segment_id, addr, value)
    }
    fn read_memory(&mut self, segment_id: u32, addr: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.memory_read(segment_id, addr)
    }
    fn read_table(&mut self, segment_id: u32, addr: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.memory_read_table(segment_id, addr)
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

    fn key_less(&mut self, v1: &Self::CellRef, i1: &Self::CellRef, v2: &Self::CellRef, i2: &Self::CellRef, index_bits: u32) -> Result<Self::CellRef, ProvingError> {
        self.key_less_by_packing(v1, i1, v2, i2, index_bits)
    }
    fn pack_key(&mut self, v: &Self::CellRef, i: &Self::CellRef, index_bits: u32) -> Result<Self::CellRef, ProvingError> {
        self.packed_key(v, i, index_bits)
    }
    fn packed_key_less(&mut self, k1: &Self::CellRef, k2: &Self::CellRef, _index_bits: u32) -> Result<Self::CellRef, ProvingError> {
        self.less_of_packed_keys(k1, k2)
    }
    fn packed_key_index(&mut self, key: &Self::CellRef, index_bits: u32) -> Result<Self::CellRef, ProvingError> {
        self.index_of_packed_key(key, index_bits)
    }

    fn identity_check(&mut self, identity: crate::ir_defs::Identity, operands: &[Self::CellRef]) -> Result<(), ProvingError> {
        self.check_identity(identity, operands)
    }

    /// `floor(raw / 2^P)`: the rescale of `fixed_product_sum`.
    fn rescale_raw_f(&mut self, raw: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        self.rescale_by(raw, self.params.precision_bits)
    }

    /// A free cell holding the hint's lane (`kernel::hint_lane` of the
    /// operands' values; placeholder values at key generation). Loaded by
    /// [`Self::hint`], so the tamper tests reach it; the IR pins it.
    fn prover_hint(&mut self, kind: crate::ir_defs::HintKind, operands: &[Self::CellRef], rank: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        let values: Vec<Fp> = operands.iter().map(|c| c.value).collect();
        let lane = kernel::hint_lane(kind, &values, rank.value);
        if self.plan.bit_hints.contains(&self.current_stmt) {
            return Ok(self.hint_in_bit_gate(lane));
        }
        Ok(self.hint(lane))
    }
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

#[cfg(test)]
pub(crate) mod two_phase_tests;
