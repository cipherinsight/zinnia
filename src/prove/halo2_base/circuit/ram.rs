//! Read-write memory by a memory-consistency argument: the RAM traces of
//! segments written at runtime addresses.
//!
//! A segment in RAM mode (`synthesizer/memory.rs`, chosen per segment by
//! cost in `synthesizer/plan.rs`) records, from its first write at a runtime
//! address on, every access in program order: first one *initial row* per
//! slot `(time 0, write, address j, the slot's value)`, then each write
//! `(t, write, address, value)` and each read `(t, read, address, value)`,
//! with `t = 1, 2, ...` in program order. Time and kind are fixed columns;
//! an access's address and value are advice cells copy-constrained to the
//! program's cells (an initial row's address is a fixed column). A read's
//! value is a hint.
//!
//! The memories of `IR::RamInit` share one trace (`synthesizer/ram.rs`):
//! memory `t` owns the addresses `[base_t, base_t + n_t)`, and an access's
//! row holds the program's address cell `a` with the fixed base, the
//! address being `base_t + a`. The trace proves every address in `[0, N)`
//! for the total `N`; that `a` lies in `[0, n_t)`, so the access stays in
//! its own memory, is the program's index check (the lowering's
//! precondition, `lower::memory::versions`).
//!
//! The prover also lays out the same rows sorted by (address, time): the
//! *sorted trace*, advice. halo2's shuffle argument proves the two traces
//! equal as multisets of `(enabled, time, kind, address, value)` (the
//! enable component keeps the unused rows of either side from standing in
//! for real ones). Gates on the sorted trace, row `r` against row `r - 1`
//! (`step = address_r - address_{r-1}`, `same = 1 - step`):
//!
//! * `step · same = 0`: the address stays or grows by one;
//! * the first row's address is 0 and the last row's is `n - 1`;
//! * `gap = same · (time_r - time_{r-1} - 1)` is looked up in the fixed
//!   column `0..len`: within one address the time strictly increases;
//! * `same_read = same · read_r` and `same_read · (value_r - value_{r-1}) = 0`:
//!   a read has the value of the row before it at its address.
//!
//! **Soundness.** The sorted addresses run from 0 to `n - 1` in steps of 0
//! or 1, so every row of the multiset, hence every access, has an address in
//! `[0, n)`: an out-of-range access has no witness. Every address has
//! exactly one initial row, at time 0, the smallest time; within an address
//! the times strictly increase, so the sorted rows of an address are its
//! initial row and then its accesses in program order (times are distinct
//! per access). A read equals the row before it, which is (by induction) the
//! value of the last write before it, or the slot's initial value: what the
//! program reads. The sorted trace is unique given the multiset (sorted by
//! a key that is distinct per row), so every read value is pinned. The
//! shuffle argument's challenges are drawn after the advice is committed:
//! the same Fiat-Shamir assumption as halo2's lookups.
//!
//! **Completeness.** The honest prover writes the program's read values and
//! sorts the trace; gaps are below the trace length, which the gap column
//! holds.
//!
//! **Cost.** Per row: two advice cells in program order (address, value; an
//! initial row only the value) and six in the sorted trace. Per segment: 8
//! advice columns, one shuffle and one lookup argument (`estimate_params`
//! counts them); the trace must fit the usable rows.

use halo2_base::halo2_proofs::{
    circuit::{Layouter, Value},
    halo2curves::bn256::Fr as Fp,
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, VirtualCells},
    poly::Rotation,
};
use halo2_base::utils::halo2::{constrain_virtual_equals_external, raw_assign_advice, raw_assign_fixed};
use halo2_base::utils::ScalarField;
use halo2_base::virtual_region::copy_constraints::SharedCopyConstraintManager;
use halo2_base::AssignedValue;

/// One row of a trace in program order.
#[derive(Clone)]
pub struct RamRow {
    pub time: u64,
    pub read: bool,
    /// A fixed address (an initial row) or the access's address cell.
    pub address: RamAddress,
    pub value: AssignedValue<Fp>,
}

/// The address of a row: fixed, or a cell plus a fixed base (several
/// memories share one trace, each in its own range of addresses; a cell
/// holds the address within its memory).
#[derive(Clone)]
pub enum RamAddress {
    Fixed(u64),
    Cell { base: u64, cell: AssignedValue<Fp> },
}

/// One segment's trace: `n` slots, its rows in program order.
#[derive(Clone)]
pub struct RamTrace {
    pub slots: u64,
    pub rows: Vec<RamRow>,
}

/// Every RAM trace of a circuit.
#[derive(Clone, Default)]
pub struct Ram {
    pub traces: Vec<RamTrace>,
}

impl Ram {

    /// The rows the longest trace needs (each trace has its own columns).
    pub fn rows_needed(&self) -> usize {
        self.traces.iter().map(|t| t.rows.len()).max().unwrap_or(0)
    }

    /// Advice cells: per row two in program order (one for an initial row)
    /// and six sorted.
    pub fn advice_cells(&self) -> usize {
        self.traces
            .iter()
            .flat_map(|t| &t.rows)
            .map(|r| if matches!(r.address, RamAddress::Fixed(_)) { 7 } else { 8 })
            .sum()
    }
}

impl RamRow {
    fn address_value(&self) -> Fp {
        match &self.address {
            RamAddress::Fixed(a) => Fp::from(*a),
            RamAddress::Cell { base, cell } => Fp::from(*base) + *cell.value(),
        }
    }
}

/// The columns, gates and arguments of one trace.
#[derive(Clone, Debug)]
struct TraceConfig {
    // program order
    p_en: Column<Fixed>,
    p_time: Column<Fixed>,
    p_read: Column<Fixed>,
    p_addr_fixed: Column<Fixed>,
    p_dyn: Column<Fixed>,
    p_addr: Column<Advice>,
    p_val: Column<Advice>,
    // sorted
    s_en: Column<Fixed>,
    q_adj: Column<Fixed>,
    q_first: Column<Fixed>,
    q_last: Column<Fixed>,
    last_addr: Column<Fixed>,
    gap_table: Column<Fixed>,
    s_time: Column<Advice>,
    s_read: Column<Advice>,
    s_addr: Column<Advice>,
    s_val: Column<Advice>,
    s_same_read: Column<Advice>,
    s_gap: Column<Advice>,
}

/// The configuration of every trace (one column set each).
#[derive(Clone, Debug)]
pub struct RamConfig {
    traces: Vec<TraceConfig>,
}

impl RamConfig {
    pub fn configure(meta: &mut ConstraintSystem<Fp>, traces: usize) -> Self {
        RamConfig { traces: (0..traces).map(|_| TraceConfig::configure(meta)).collect() }
    }

    /// Lays out every trace: the program order as recorded, the sorted
    /// trace computed from the cells' values. With `copy` (keygen and mock)
    /// the program-order address and value cells are constrained to the
    /// program's cells.
    pub fn assign(&self, ram: &Ram, layouter: &mut impl Layouter<Fp>, copy: Option<&SharedCopyConstraintManager<Fp>>) -> Result<(), Error> {
        assert_eq!(ram.traces.len(), self.traces.len(), "the pinned RAM layout");
        for (trace, config) in ram.traces.iter().zip(&self.traces) {
            config.assign(trace, layouter, copy)?;
        }
        Ok(())
    }
}

impl TraceConfig {
    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self {
        let mut fixed = || meta.fixed_column();
        let (p_en, p_time, p_read, p_addr_fixed, p_dyn) = (fixed(), fixed(), fixed(), fixed(), fixed());
        let (s_en, q_adj, q_first, q_last, last_addr, gap_table) = (fixed(), fixed(), fixed(), fixed(), fixed(), fixed());
        let mut advice = || meta.advice_column();
        let (p_addr, p_val) = (advice(), advice());
        let (s_time, s_read, s_addr, s_val, s_same_read, s_gap) = (advice(), advice(), advice(), advice(), advice(), advice());
        meta.enable_equality(p_addr);
        meta.enable_equality(p_val);
        meta.create_gate("RAM sorted trace", |m| {
            let q_adj = m.query_fixed(q_adj, Rotation::cur());
            let q_first = m.query_fixed(q_first, Rotation::cur());
            let q_last = m.query_fixed(q_last, Rotation::cur());
            let last_addr = m.query_fixed(last_addr, Rotation::cur());
            let addr = m.query_advice(s_addr, Rotation::cur());
            let addr_prev = m.query_advice(s_addr, Rotation::prev());
            let time = m.query_advice(s_time, Rotation::cur());
            let time_prev = m.query_advice(s_time, Rotation::prev());
            let read = m.query_advice(s_read, Rotation::cur());
            let val = m.query_advice(s_val, Rotation::cur());
            let val_prev = m.query_advice(s_val, Rotation::prev());
            let same_read = m.query_advice(s_same_read, Rotation::cur());
            let gap = m.query_advice(s_gap, Rotation::cur());
            let one = Expression::Constant(Fp::one());
            let step = addr.clone() - addr_prev;
            let same = one.clone() - step.clone();
            vec![
                q_adj.clone() * step * same.clone(),
                q_adj.clone() * (same_read.clone() - same.clone() * read),
                q_adj.clone() * same_read * (val - val_prev),
                q_adj * (gap - same * (time - time_prev - one)),
                q_first * addr.clone(),
                q_last * (addr - last_addr),
            ]
        });
        meta.lookup_any("RAM time gap", |m| {
            vec![(m.query_advice(s_gap, Rotation::cur()), m.query_fixed(gap_table, Rotation::cur()))]
        });
        // The sorted trace is the shuffle's "input" side and the program
        // order its "shuffle" side (multiset equality is symmetric): the
        // pinned halo2_proofs (PSE v0.3.0) sizes the input side's evaluation
        // buffer from the shuffle side's expressions (`evaluation.rs`,
        // `shuffle_evaluator.instance()` for both), so the side with more
        // intermediate values (the program order's address expression) must
        // be the shuffle side, or the real prover indexes out of bounds.
        meta.shuffle("RAM trace", |m: &mut VirtualCells<Fp>| {
            let fx = |m: &mut VirtualCells<Fp>, c| m.query_fixed(c, Rotation::cur());
            let ad = |m: &mut VirtualCells<Fp>, c| m.query_advice(c, Rotation::cur());
            let p_addr_expr = fx(m, p_addr_fixed) + fx(m, p_dyn) * ad(m, p_addr);
            vec![
                (fx(m, s_en), fx(m, p_en)),
                (ad(m, s_time), fx(m, p_time)),
                (ad(m, s_read), fx(m, p_read)),
                (ad(m, s_addr), p_addr_expr),
                (ad(m, s_val), ad(m, p_val)),
            ]
        });
        TraceConfig {
            p_en, p_time, p_read, p_addr_fixed, p_dyn, p_addr, p_val, s_en, q_adj, q_first, q_last, last_addr, gap_table,
            s_time, s_read, s_addr, s_val, s_same_read, s_gap,
        }
    }

    fn assign(&self, trace: &RamTrace, layouter: &mut impl Layouter<Fp>, copy: Option<&SharedCopyConstraintManager<Fp>>) -> Result<(), Error> {
        layouter.assign_region(
            || "RAM trace",
            |mut region| {
                let mut copy = copy.map(|c| c.lock().unwrap());
                let mut place = |region: &mut _, column, row, cell: &AssignedValue<Fp>| {
                    let raw = raw_assign_advice(region, column, row, Value::known(*cell.value()));
                    if let Some(copy) = copy.as_mut() {
                        constrain_virtual_equals_external(region, *cell, raw.cell(), copy);
                    }
                };
                for (row, r) in trace.rows.iter().enumerate() {
                    raw_assign_fixed(&mut region, self.p_en, row, Fp::one());
                    raw_assign_fixed(&mut region, self.p_time, row, Fp::from(r.time));
                    raw_assign_fixed(&mut region, self.p_read, row, Fp::from(r.read as u64));
                    match &r.address {
                        RamAddress::Fixed(a) => {
                            raw_assign_fixed(&mut region, self.p_addr_fixed, row, Fp::from(*a));
                        }
                        RamAddress::Cell { base, cell } => {
                            raw_assign_fixed(&mut region, self.p_addr_fixed, row, Fp::from(*base));
                            raw_assign_fixed(&mut region, self.p_dyn, row, Fp::one());
                            place(&mut region, self.p_addr, row, cell);
                        }
                    }
                    place(&mut region, self.p_val, row, &r.value);
                }
                // The sorted trace: by address, then time (the values the
                // prover holds; witness-free keygen sorts zeros, which only
                // the advice depends on).
                let mut sorted: Vec<(u64, u64, bool, Fp)> = trace
                    .rows
                    .iter()
                    .map(|r| (r.address_value().get_lower_64(), r.time, r.read, *r.value.value()))
                    .collect();
                sorted.sort_by_key(|&(a, t, _, _)| (a, t));
                let len = sorted.len();
                for (row, &(addr, time, read, value)) in sorted.iter().enumerate() {
                    raw_assign_fixed(&mut region, self.s_en, row, Fp::one());
                    raw_assign_fixed(&mut region, self.gap_table, row, Fp::from(row as u64));
                    if row == 0 {
                        raw_assign_fixed(&mut region, self.q_first, row, Fp::one());
                    } else {
                        raw_assign_fixed(&mut region, self.q_adj, row, Fp::one());
                    }
                    if row + 1 == len {
                        raw_assign_fixed(&mut region, self.q_last, row, Fp::one());
                        raw_assign_fixed(&mut region, self.last_addr, row, Fp::from(trace.slots - 1));
                    }
                    let (same, gap) = match row.checked_sub(1).map(|p| sorted[p]) {
                        Some((prev_addr, prev_time, _, _)) if prev_addr == addr => (true, time.saturating_sub(prev_time + 1)),
                        _ => (false, 0),
                    };
                    raw_assign_advice(&mut region, self.s_time, row, Value::known(Fp::from(time)));
                    raw_assign_advice(&mut region, self.s_read, row, Value::known(Fp::from(read as u64)));
                    raw_assign_advice(&mut region, self.s_addr, row, Value::known(Fp::from(addr)));
                    raw_assign_advice(&mut region, self.s_val, row, Value::known(value));
                    raw_assign_advice(&mut region, self.s_same_read, row, Value::known(Fp::from((same && read) as u64)));
                    raw_assign_advice(&mut region, self.s_gap, row, Value::known(Fp::from(gap)));
                }
                Ok(())
            },
        )
    }
}
