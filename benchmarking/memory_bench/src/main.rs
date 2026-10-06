//! Per-access cost of memory constructions in halo2-base (layer-memory, phase 1).
//!
//! Workloads (all inputs private witnesses):
//!
//! * `read <mode> n d c`: `c` arrays of `n` values, `d` private indices, each
//!   index read from every array (`r = d·c` reads; the `c` reads at one index
//!   share whatever a construction builds per index).
//! * `write <mode> n w r`: one array of `n` values, `w` writes of private
//!   values at private indices, then `r` reads at private indices.
//!
//! Constructions (`mode`):
//!
//! * `onehot`: `[i == j]` per (index, slot) shared by the arrays, exactly-one
//!   asserted (it proves `0 <= i < n`), one inner product per read.
//! * `chain`: the same indicators, a select chain per read.
//! * `mux`: `i < n` by a range check, `i`'s bits, a binary select tree per read.
//! * `rom`: a dynamic lookup (`lookup_any`) into a table `(tag, j, a_j)` whose
//!   tag and address are fixed columns and whose value column is
//!   copy-constrained to the array cells; a read is a pair `(i, v)` of
//!   witnesses that must equal a row of its table. Proves `v = a_i` and
//!   `0 <= i < n`: the table's addresses are exactly `0..n`.
//! * `overlay` (writes): reads by one-hot on the initial array, then every
//!   earlier write applied in program order (`[w_addr == r_addr]`, select).
//!   This is Zinnia's halo2-base lowering today.
//! * `eager` (writes): every write updates every slot (`[w == j]`, select).
//! * `ram` (writes): a memory-consistency argument. The program-order trace
//!   (time and kind fixed, address and value advice copied from the
//!   program's cells; `n` initial rows at time 0) and a sorted trace (advice,
//!   the prover's order by address then time) are equal as multisets by
//!   halo2's shuffle argument (challenges theta and gamma, drawn after the
//!   advice is committed). Sorted-trace gates: the address steps by 0 or 1
//!   from 0 to `n - 1` (so every access address is in `[0, n)`); within one
//!   address the time strictly increases (`t' - t - 1` in the range table);
//!   a read row has the value of the row before it. Every address has its
//!   initial row at time 0, which therefore comes first, so a read sees the
//!   last earlier write. The sorted trace is unique (times are distinct per
//!   address), and so is every read value.
//!
//! Output: one JSON line with the cells (advice incl. memory columns + range
//! lookup cells), the input floor, `k`, the committed polynomials and their
//! cost `polys · 2^k` (Zinnia's `estimate_params` model), and with `--prove`
//! proving time; `--tamper` runs MockProver on a cheating witness (read 0's
//! value + 1) and reports whether it is accepted.

use std::time::Instant;

use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, BaseConfig, CircuitBuilderStage, MaybeRangeConfig};
use halo2_base::gates::{GateInstructions, RangeInstructions};
use halo2_base::halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    halo2curves::bn256::{Fr, G1Affine},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column, ConstraintSystem, Error, Expression,
        Fixed,
    },
    poly::{
        commitment::ParamsProver,
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::{ProverIPA, VerifierIPA},
            strategy::SingleStrategy,
        },
        Rotation,
    },
    transcript::{Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use halo2_base::utils::halo2::{constrain_virtual_equals_external, raw_assign_advice, raw_assign_fixed};
use halo2_base::virtual_region::manager::VirtualRegionManager;
use halo2_base::halo2_proofs::poly::VerificationStrategy;
use halo2_base::utils::ScalarField;
use halo2_base::{AssignedValue, QuantumCell};
use rand::rngs::OsRng;
use rand::{Rng, SeedableRng};

const LOOKUP_BITS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    OneHot,
    Chain,
    Mux,
    Rom,
    Overlay,
    Eager,
    Ram,
}

#[derive(Clone)]
struct Workload {
    /// `arrays[a][j]`.
    arrays: Vec<Vec<u64>>,
    /// Read workload: indices, each read from every array.
    read_idx: Vec<usize>,
    /// Write workload (array 0): writes, then reads.
    writes: Vec<(usize, u64)>,
    reads_after: Vec<usize>,
}

/// One row of the ROM table or of a lookup: `(tag, address, value)`.
struct RomRow {
    tag: u64,
    addr: u64,
    value: AssignedValue<Fr>,
}
struct RomKey {
    tag: u64,
    idx: AssignedValue<Fr>,
    value: AssignedValue<Fr>,
}
/// One program-order trace row: time and kind fixed, address fixed (initial
/// rows) or a cell (accesses), value a cell.
struct TraceRow {
    time: u64,
    is_read: bool,
    addr: Result<u64, AssignedValue<Fr>>,
    value: AssignedValue<Fr>,
}

#[derive(Default)]
struct Memory {
    rom_rows: Vec<RomRow>,
    rom_keys: Vec<RomKey>,
    trace: Vec<TraceRow>,
    n: usize,
}

#[derive(Clone, Debug)]
struct RomConfig {
    t_tag: Column<Fixed>,
    t_addr: Column<Fixed>,
    t_en: Column<Fixed>,
    t_val: Column<Advice>,
    k_tag: Column<Fixed>,
    k_en: Column<Fixed>,
    k_idx: Column<Advice>,
    k_val: Column<Advice>,
}

impl RomConfig {
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let (t_tag, t_addr, t_en, k_tag, k_en) =
            (meta.fixed_column(), meta.fixed_column(), meta.fixed_column(), meta.fixed_column(), meta.fixed_column());
        let (t_val, k_idx, k_val) = (meta.advice_column(), meta.advice_column(), meta.advice_column());
        for c in [t_val, k_idx, k_val] {
            meta.enable_equality(c);
        }
        meta.lookup_any("rom", |m| {
            let q = |c: Column<Fixed>, m: &mut halo2_base::halo2_proofs::plonk::VirtualCells<Fr>| m.query_fixed(c, Rotation::cur());
            let a = |c: Column<Advice>, m: &mut halo2_base::halo2_proofs::plonk::VirtualCells<Fr>| m.query_advice(c, Rotation::cur());
            vec![
                (q(k_en, m), q(t_en, m)),
                (q(k_tag, m), q(t_tag, m)),
                (a(k_idx, m), q(t_addr, m)),
                (a(k_val, m), a(t_val, m)),
            ]
        });
        Self { t_tag, t_addr, t_en, t_val, k_tag, k_en, k_idx, k_val }
    }
}

#[derive(Clone, Debug)]
struct RamConfig {
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
    s_time: Column<Advice>,
    s_read: Column<Advice>,
    s_addr: Column<Advice>,
    s_val: Column<Advice>,
    s_same_read: Column<Advice>,
    s_gap: Column<Advice>,
    gap_table: Column<Fixed>,
}

impl RamConfig {
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let f = |meta: &mut ConstraintSystem<Fr>| meta.fixed_column();
        let (p_en, p_time, p_read, p_addr_fixed, p_dyn) = (f(meta), f(meta), f(meta), f(meta), f(meta));
        let (s_en, q_adj, q_first, q_last, last_addr) = (f(meta), f(meta), f(meta), f(meta), f(meta));
        let a = |meta: &mut ConstraintSystem<Fr>| meta.advice_column();
        let (p_addr, p_val) = (a(meta), a(meta));
        let (s_time, s_read, s_addr, s_val, s_same_read, s_gap) = (a(meta), a(meta), a(meta), a(meta), a(meta), a(meta));
        meta.enable_equality(p_addr);
        meta.enable_equality(p_val);
        meta.create_gate("ram sorted trace", |m| {
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
            let one = Expression::Constant(Fr::one());
            let step = addr.clone() - addr_prev;
            let same = one.clone() - step.clone();
            vec![
                q_adj.clone() * step.clone() * same.clone(),
                q_adj.clone() * (same_read.clone() - same.clone() * read),
                q_adj.clone() * same_read * (val - val_prev),
                q_adj * (gap - same * (time - time_prev - one)),
                q_first * addr.clone(),
                q_last * (addr - last_addr),
            ]
        });
        // `gap = t - t' - 1 >= 0` within an address: a lookup into the fixed
        // column `0..len` (times are below the trace length).
        let gap_table = meta.fixed_column();
        meta.lookup_any("ram time gap", |m| {
            vec![(m.query_advice(s_gap, Rotation::cur()), m.query_fixed(gap_table, Rotation::cur()))]
        });
        meta.shuffle("ram trace", |m| {
            let fx = |m: &mut halo2_base::halo2_proofs::plonk::VirtualCells<Fr>, c| m.query_fixed(c, Rotation::cur());
            let ad = |m: &mut halo2_base::halo2_proofs::plonk::VirtualCells<Fr>, c| m.query_advice(c, Rotation::cur());
            let p_addr_expr = fx(m, p_addr_fixed) + fx(m, p_dyn) * ad(m, p_addr);
            // Sorted trace as the input side: the pinned halo2_proofs sizes
            // the input side's evaluation buffer from the shuffle side.
            vec![
                (fx(m, s_en), fx(m, p_en)),
                (ad(m, s_time), fx(m, p_time)),
                (ad(m, s_read), fx(m, p_read)),
                (ad(m, s_addr), p_addr_expr),
                (ad(m, s_val), ad(m, p_val)),
            ]
        });
        Self {
            p_en, p_time, p_read, p_addr_fixed, p_dyn, p_addr, p_val, s_en, q_adj, q_first, q_last, last_addr, s_time,
            s_read, s_addr, s_val, s_same_read, s_gap, gap_table,
        }
    }
}

#[derive(Clone, Default)]
struct BenchParams {
    base: BaseCircuitParams,
    rom: bool,
    ram: bool,
}

#[derive(Clone)]
struct BenchConfig {
    base: BaseConfig<Fr>,
    rom: Option<RomConfig>,
    ram: Option<RamConfig>,
}

struct BenchCircuit {
    builder: BaseCircuitBuilder<Fr>,
    memory: Memory,
    params: BenchParams,
}

fn fr(v: u64) -> Fr {
    Fr::from(v)
}

/// Builds the circuit's cells for `mode` on `w`; `tamper` adds 1 to the
/// value read 0 returns (a prover claiming a wrong value).
fn build(mode: Mode, w: &Workload, tamper: bool) -> (BaseCircuitBuilder<Fr>, Memory, usize) {
    let mut builder = BaseCircuitBuilder::<Fr>::from_stage(CircuitBuilderStage::Mock).use_lookup_bits(LOOKUP_BITS);
    let range = builder.range_chip();
    let gate = range.gate().clone();
    let mut memory = Memory::default();
    let ctx = builder.main(0);
    let arrays: Vec<Vec<AssignedValue<Fr>>> =
        w.arrays.iter().map(|a| ctx.assign_witnesses(a.iter().map(|v| fr(*v)))).collect();
    let n = w.arrays[0].len();
    memory.n = n;
    let mut floor = arrays.iter().map(Vec::len).sum::<usize>();
    let mut outputs: Vec<AssignedValue<Fr>> = Vec::new();
    let tamper_value = if tamper { Fr::one() } else { Fr::zero() };
    let mut first_read = true;
    let mut bump = |v: Fr| {
        let out = if first_read { v + tamper_value } else { v };
        first_read = false;
        out
    };
    if !w.read_idx.is_empty() {
        let idx = ctx.assign_witnesses(w.read_idx.iter().map(|i| fr(*i as u64)));
        floor += idx.len();
        match mode {
            Mode::OneHot | Mode::Chain => {
                for (k, i) in idx.iter().enumerate() {
                    let ind: Vec<AssignedValue<Fr>> =
                        (0..n).map(|j| gate.is_equal(ctx, *i, QuantumCell::Constant(fr(j as u64)))).collect();
                    let hits = gate.sum(ctx, ind.iter().copied());
                    gate.assert_is_const(ctx, &hits, &Fr::one());
                    for a in &arrays {
                        let out = if mode == Mode::OneHot {
                            gate.inner_product(ctx, a.iter().copied(), ind.iter().copied().map(QuantumCell::Existing))
                        } else {
                            let mut acc = a[0];
                            for j in 1..n {
                                acc = gate.select(ctx, a[j], acc, ind[j]);
                            }
                            acc
                        };
                        let _ = k;
                        outputs.push(out);
                    }
                }
            }
            Mode::Mux => {
                let width = (usize::BITS - (n - 1).leading_zeros()).max(1) as usize;
                for i in &idx {
                    range.check_less_than_safe(ctx, *i, n as u64);
                    let bits = gate.num_to_bits(ctx, *i, width);
                    for a in &arrays {
                        let mut level: Vec<AssignedValue<Fr>> = a.clone();
                        for bit in &bits {
                            level = level
                                .chunks(2)
                                .map(|p| if p.len() == 2 { gate.select(ctx, p[1], p[0], *bit) } else { p[0] })
                                .collect();
                        }
                        outputs.push(level[0]);
                    }
                }
            }
            Mode::Rom => {
                for (tag, a) in arrays.iter().enumerate() {
                    for (j, v) in a.iter().enumerate() {
                        memory.rom_rows.push(RomRow { tag: tag as u64, addr: j as u64, value: *v });
                    }
                }
                for (k, i) in idx.iter().enumerate() {
                    for (tag, a) in arrays.iter().enumerate() {
                        let v = ctx.load_witness(bump(*a[w.read_idx[k]].value()));
                        memory.rom_keys.push(RomKey { tag: tag as u64, idx: *i, value: v });
                        outputs.push(v);
                    }
                }
            }
            _ => panic!("{mode:?} is a write mode"),
        }
    } else {
        let a = &arrays[0];
        let waddr = ctx.assign_witnesses(w.writes.iter().map(|(i, _)| fr(*i as u64)));
        let wval = ctx.assign_witnesses(w.writes.iter().map(|(_, v)| fr(*v)));
        let raddr = ctx.assign_witnesses(w.reads_after.iter().map(|i| fr(*i as u64)));
        floor += waddr.len() + wval.len() + raddr.len();
        let mut current: Vec<u64> = w.arrays[0].clone();
        for (i, v) in &w.writes {
            current[*i] = *v;
        }
        match mode {
            Mode::Overlay => {
                for (k, r) in raddr.iter().enumerate() {
                    let ind: Vec<AssignedValue<Fr>> =
                        (0..n).map(|j| gate.is_equal(ctx, *r, QuantumCell::Constant(fr(j as u64)))).collect();
                    let hits = gate.sum(ctx, ind.iter().copied());
                    gate.assert_is_const(ctx, &hits, &Fr::one());
                    let mut out = gate.inner_product(ctx, a.iter().copied(), ind.iter().copied().map(QuantumCell::Existing));
                    for (wa, wv) in waddr.iter().zip(&wval) {
                        let hit = gate.is_equal(ctx, *wa, *r);
                        out = gate.select(ctx, *wv, out, hit);
                    }
                    let _ = k;
                    outputs.push(out);
                }
                for wa in &waddr {
                    range.check_less_than_safe(ctx, *wa, n as u64);
                }
            }
            Mode::Eager => {
                let mut slots = a.clone();
                for (wa, wv) in waddr.iter().zip(&wval) {
                    let ind: Vec<AssignedValue<Fr>> =
                        (0..n).map(|j| gate.is_equal(ctx, *wa, QuantumCell::Constant(fr(j as u64)))).collect();
                    let hits = gate.sum(ctx, ind.iter().copied());
                    gate.assert_is_const(ctx, &hits, &Fr::one());
                    for j in 0..n {
                        slots[j] = gate.select(ctx, *wv, slots[j], ind[j]);
                    }
                }
                for r in &raddr {
                    let ind: Vec<AssignedValue<Fr>> =
                        (0..n).map(|j| gate.is_equal(ctx, *r, QuantumCell::Constant(fr(j as u64)))).collect();
                    let hits = gate.sum(ctx, ind.iter().copied());
                    gate.assert_is_const(ctx, &hits, &Fr::one());
                    outputs.push(gate.inner_product(ctx, slots.iter().copied(), ind.iter().copied().map(QuantumCell::Existing)));
                }
            }
            Mode::Ram => {
                for (j, v) in a.iter().enumerate() {
                    memory.trace.push(TraceRow { time: 0, is_read: false, addr: Ok(j as u64), value: *v });
                }
                let mut time = 1;
                for (wa, wv) in waddr.iter().zip(&wval) {
                    memory.trace.push(TraceRow { time, is_read: false, addr: Err(*wa), value: *wv });
                    time += 1;
                }
                for (k, r) in raddr.iter().enumerate() {
                    let v = ctx.load_witness(bump(fr(current[w.reads_after[k]])));
                    memory.trace.push(TraceRow { time, is_read: true, addr: Err(*r), value: v });
                    outputs.push(v);
                    time += 1;
                }
            }
            _ => panic!("{mode:?} is a read mode"),
        }
    }
    // Consume the outputs (a sum), so nothing is dead.
    let total = gate.sum(ctx, outputs.iter().copied());
    let _ = total;
    let _ = &mut bump;
    (builder, memory, floor)
}

impl Circuit<Fr> for BenchCircuit {
    type Config = BenchConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = BenchParams;

    fn params(&self) -> BenchParams {
        self.params.clone()
    }
    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }
    fn configure(_: &mut ConstraintSystem<Fr>) -> BenchConfig {
        unreachable!()
    }
    fn configure_with_params(meta: &mut ConstraintSystem<Fr>, p: BenchParams) -> BenchConfig {
        let base = BaseConfig::configure(meta, p.base.clone());
        let rom = p.rom.then(|| RomConfig::configure(meta));
        let ram = p.ram.then(|| RamConfig::configure(meta));
        let mut config = BenchConfig { base, rom, ram };
        let usable = (1usize << p.base.k) - meta.minimum_rows();
        config.base.set_usable_rows(usable);
        config
    }

    fn synthesize(&self, config: BenchConfig, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
        config.base.initialize(&mut layouter);
        let core = self.builder.core();
        layouter.assign_region(
            || "core",
            |mut region| {
                let usable = config.base.gate().max_rows;
                core.phase_manager[0].assign_raw(&(config.base.gate().basic_gates[0].clone(), usable), &mut region);
                if let MaybeRangeConfig::WithRange(range) = &config.base.base {
                    self.builder.assign_lookups_in_phase(range, &mut region, 0);
                }
                Ok(())
            },
        )?;
        let copy = &core.copy_manager;
        if let Some(rom) = &config.rom {
            layouter.assign_region(
                || "rom",
                |mut region| {
                    let mut cm = copy.lock().unwrap();
                    for (row, t) in self.memory.rom_rows.iter().enumerate() {
                        raw_assign_fixed(&mut region, rom.t_en, row, Fr::one());
                        raw_assign_fixed(&mut region, rom.t_tag, row, fr(t.tag));
                        raw_assign_fixed(&mut region, rom.t_addr, row, fr(t.addr));
                        let c = raw_assign_advice(&mut region, rom.t_val, row, Value::known(*t.value.value()));
                        constrain_virtual_equals_external(&mut region, t.value, c.cell(), &mut cm);
                    }
                    for (row, key) in self.memory.rom_keys.iter().enumerate() {
                        raw_assign_fixed(&mut region, rom.k_en, row, Fr::one());
                        raw_assign_fixed(&mut region, rom.k_tag, row, fr(key.tag));
                        let c = raw_assign_advice(&mut region, rom.k_idx, row, Value::known(*key.idx.value()));
                        constrain_virtual_equals_external(&mut region, key.idx, c.cell(), &mut cm);
                        let c = raw_assign_advice(&mut region, rom.k_val, row, Value::known(*key.value.value()));
                        constrain_virtual_equals_external(&mut region, key.value, c.cell(), &mut cm);
                    }
                    Ok(())
                },
            )?;
        }
        if let Some(ram) = &config.ram {
            layouter.assign_region(
                || "ram",
                |mut region| {
                    let mut cm = copy.lock().unwrap();
                    // program order
                    for (row, t) in self.memory.trace.iter().enumerate() {
                        raw_assign_fixed(&mut region, ram.p_en, row, Fr::one());
                        raw_assign_fixed(&mut region, ram.p_time, row, fr(t.time));
                        raw_assign_fixed(&mut region, ram.p_read, row, fr(t.is_read as u64));
                        match &t.addr {
                            Ok(a) => {
                                raw_assign_fixed(&mut region, ram.p_addr_fixed, row, fr(*a));
                            }
                            Err(cell) => {
                                raw_assign_fixed(&mut region, ram.p_dyn, row, Fr::one());
                                let c = raw_assign_advice(&mut region, ram.p_addr, row, Value::known(*cell.value()));
                                constrain_virtual_equals_external(&mut region, *cell, c.cell(), &mut cm);
                            }
                        }
                        let c = raw_assign_advice(&mut region, ram.p_val, row, Value::known(*t.value.value()));
                        constrain_virtual_equals_external(&mut region, t.value, c.cell(), &mut cm);
                    }
                    // sorted: the prover's order (address, then time)
                    let mut sorted: Vec<(u64, u64, bool, Fr)> = self
                        .memory
                        .trace
                        .iter()
                        .map(|t| {
                            let a = match &t.addr {
                                Ok(a) => *a,
                                Err(c) => c.value().get_lower_64(),
                            };
                            (a, t.time, t.is_read, *t.value.value())
                        })
                        .collect();
                    sorted.sort_by_key(|(a, t, _, _)| (*a, *t));
                    let len = sorted.len();
                    for (row, (a, t, read, v)) in sorted.iter().enumerate() {
                        raw_assign_fixed(&mut region, ram.s_en, row, Fr::one());
                        raw_assign_fixed(&mut region, ram.gap_table, row, fr(row as u64));
                        if row == 0 {
                            raw_assign_fixed(&mut region, ram.q_first, row, Fr::one());
                        } else {
                            raw_assign_fixed(&mut region, ram.q_adj, row, Fr::one());
                        }
                        if row + 1 == len {
                            raw_assign_fixed(&mut region, ram.q_last, row, Fr::one());
                            raw_assign_fixed(&mut region, ram.last_addr, row, fr(self.memory.n as u64 - 1));
                        }
                        raw_assign_advice(&mut region, ram.s_time, row, Value::known(fr(*t)));
                        raw_assign_advice(&mut region, ram.s_read, row, Value::known(fr(*read as u64)));
                        raw_assign_advice(&mut region, ram.s_addr, row, Value::known(fr(*a)));
                        raw_assign_advice(&mut region, ram.s_val, row, Value::known(*v));
                        let (same, gap) = if row == 0 {
                            (false, 0)
                        } else {
                            let (pa, pt, _, _) = sorted[row - 1];
                            (pa == *a, if pa == *a { t - pt - 1 } else { 0 })
                        };
                        raw_assign_advice(&mut region, ram.s_same_read, row, Value::known(fr((same && *read) as u64)));
                        raw_assign_advice(&mut region, ram.s_gap, row, Value::known(fr(gap)));
                    }
                    Ok(())
                },
            )?;
        }
        layouter.assign_region(
            || "copy",
            |mut region| {
                copy.assign_raw(config.base.constants(), &mut region);
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn workload(kind: &str, n: usize, a: usize, b: usize, rng: &mut impl Rng) -> Workload {
    match kind {
        "read" => {
            let (d, c) = (a, b);
            Workload {
                arrays: (0..c).map(|_| (0..n).map(|_| rng.gen::<u32>() as u64).collect()).collect(),
                read_idx: (0..d).map(|_| rng.gen_range(0..n)).collect(),
                writes: vec![],
                reads_after: vec![],
            }
        }
        _ => {
            let (w, r) = (a, b);
            Workload {
                arrays: vec![(0..n).map(|_| rng.gen::<u32>() as u64).collect()],
                read_idx: vec![],
                writes: (0..w).map(|_| (rng.gen_range(0..n), rng.gen::<u32>() as u64)).collect(),
                reads_after: (0..r).map(|_| rng.gen_range(0..n)).collect(),
            }
        }
    }
}

/// Memory-region rows a construction needs (they must fit the usable rows).
fn memory_rows(m: &Memory) -> usize {
    m.rom_rows.len().max(m.rom_keys.len()).max(m.trace.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args[1].clone();
    let mode = match args[2].as_str() {
        "onehot" => Mode::OneHot,
        "chain" => Mode::Chain,
        "mux" => Mode::Mux,
        "rom" => Mode::Rom,
        "overlay" => Mode::Overlay,
        "eager" => Mode::Eager,
        "ram" => Mode::Ram,
        other => panic!("mode {other}"),
    };
    let n: usize = args[3].parse().unwrap();
    let a: usize = args[4].parse().unwrap();
    let b: usize = args[5].parse().unwrap();
    let prove = args.iter().any(|x| x == "--prove");
    let tamper = args.iter().any(|x| x == "--tamper");
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let w = workload(&kind, n, a, b, &mut rng);

    let (mut builder, memory, floor) = build(mode, &w, false);
    let stats = builder.statistics();
    let gate_cells: usize = stats.gate.total_advice_per_phase.iter().sum();
    let lookup_cells: usize = stats.total_lookup_advice_per_phase.iter().sum();
    let mem_rows = memory_rows(&memory);
    let mem_cells = memory.rom_rows.len() + 2 * memory.rom_keys.len()
        + memory.trace.iter().map(|t| if t.addr.is_ok() { 1 } else { 2 }).sum::<usize>()
        + 6 * memory.trace.len();
    // Pick k by Zinnia's rule: minimise committed polynomials · 2^k.
    let mut best: Option<(u128, usize, BenchParams, usize)> = None;
    for k in 9..=22usize {
        let mut meta = ConstraintSystem::<Fr>::default();
        let min_rows_guess = 30;
        let usable = (1usize << k) - min_rows_guess;
        if mem_rows > usable || (1 << LOOKUP_BITS) > usable {
            continue;
        }
        builder.set_k(k);
        let base = builder.calculate_params(Some(min_rows_guess));
        let params = BenchParams { base, rom: mode == Mode::Rom, ram: mode == Mode::Ram };
        let _ = BenchCircuit::configure_with_params(&mut meta, params.clone());
        if (1usize << k) - meta.minimum_rows() < mem_rows {
            continue;
        }
        let degree = meta.degree().max(3);
        let perm_chunk = (degree - 2).max(1);
        let polys = meta.num_advice_columns()
            + 3 * meta.lookups().len()
            + meta.shuffles().len()
            + meta.permutation().get_columns().len().div_ceil(perm_chunk)
            + (degree - 1)
            + 1;
        let cost = (polys as u128) << k;
        if best.as_ref().map_or(true, |(c, ..)| cost < *c) {
            best = Some((cost, k, params, polys));
        }
    }
    let (cost, k, params, polys) = best.expect("some k fits");
    builder.set_k(k);
    builder.config_params = params.base.clone();
    let mut report = serde_json::json!({
        "kind": kind, "mode": format!("{mode:?}"), "n": n, "a": a, "b": b, "k": k,
        "gate_cells": gate_cells, "range_lookup_cells": lookup_cells, "memory_cells": mem_cells,
        "cells": gate_cells + lookup_cells + mem_cells, "floor": floor,
        "polys": polys, "poly_cost": cost as u64,
    });
    let circuit = BenchCircuit { builder, memory, params: params.clone() };
    if tamper {
        let honest = MockProver::run(k as u32, &circuit, vec![]).unwrap().verify().is_ok();
        let (mut bad_builder, bad_memory, _) = build(mode, &w, true);
        bad_builder.set_k(k);
        bad_builder.config_params = params.base.clone();
        let bad = BenchCircuit { builder: bad_builder, memory: bad_memory, params: params.clone() };
        let cheated = MockProver::run(k as u32, &bad, vec![]).unwrap().verify().is_ok();
        report["mock_honest_accepted"] = honest.into();
        report["mock_tampered_accepted"] = cheated.into();
    }
    if prove {
        let started = Instant::now();
        let ipa = ParamsIPA::<G1Affine>::new(k as u32);
        let vk = keygen_vk(&ipa, &circuit).unwrap();
        let pk = keygen_pk(&ipa, vk, &circuit).unwrap();
        report["keygen_s"] = started.elapsed().as_secs_f64().into();
        rayon::broadcast(|_| ());
        let mut samples = Vec::new();
        let mut proof = Vec::new();
        for _ in 0..3 {
            let started = Instant::now();
            let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<_>>::init(Vec::new());
            create_proof::<IPACommitmentScheme<G1Affine>, ProverIPA<G1Affine>, _, _, _, _>(
                &ipa, &pk, std::slice::from_ref(&circuit), &[&[]], OsRng, &mut transcript,
            )
            .unwrap();
            proof = transcript.finalize();
            samples.push(started.elapsed().as_secs_f64());
        }
        samples.sort_by(f64::total_cmp);
        report["prove_median_s"] = samples[1].into();
        let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
        let ok = verify_proof::<IPACommitmentScheme<G1Affine>, VerifierIPA<G1Affine>, _, _, _>(
            &ipa,
            pk.get_vk(),
            SingleStrategy::new(&ipa),
            &[&[]],
            &mut transcript,
        )
        .is_ok();
        report["verified"] = ok.into();
    }
    println!("{report}");
}
