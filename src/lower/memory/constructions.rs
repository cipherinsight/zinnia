//! The memory family's constructions for `OpKind::Read` groups.
//!
//! A group is the reads of one line of `n >= 2` cells `x[0..n)`, each at a
//! runtime index `i` (the lane's own operand). The frontend emitted the
//! program's check `0 <= i < n` for each `i`; NumPy raises otherwise, so
//! there is nothing to prove for such an `i`.
//!
//! Every construction below produces every lane of the group (one candidate
//! per group): the selection weighs whole groups. Per-index parts (equality
//! tests, index bits) are shared with reads of other lines at the same index;
//! the cost charges each read its share (`cost::index_share`), and the
//! cleanup after lowering merges the duplicate statements.

use crate::builder::IRBuilder;
use crate::lower::gather::window::{scale_by_bit, sum_typed};
use crate::lower::{Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, OpKind, Requirement, Scope};
use crate::types::{NumberType, StmtId, Value};

use super::cost::{self, ReadPlan};
use super::semantics::dtype_of;

/// The family's constructions, in the order the selection breaks ties
/// (the frontend's former default first).
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&ReadBySelectChain, &ReadByIndexBits, &ReadByOneHot, &ReadByMemory]
}

/// `0 <= i < n` on every path (the index operand is the frontend's guarded
/// index, in range where the read does not run).
fn in_bounds(index: StmtId, n: usize) -> Requirement {
    Requirement::Fact(Fact { claim: Claim::InRange { x: index, lo: 0, hi: Bound::Const(n as i64) }, scope: Scope::Everywhere })
}

fn indices(group: &Group) -> Vec<StmtId> {
    let mut indices: Vec<StmtId> = group.lanes.iter().map(|l| l.own[0]).collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// The candidate of `plan` for `group`, or none when `ZINNIA_MEMORY_READ`
/// forces another construction.
fn candidate(group: &Group, cx: &Context, plan: ReadPlan, emitter: Box<dyn Emitter>) -> Vec<Candidate> {
    if group.kind != OpKind::Read {
        return Vec::new();
    }
    if super::forced().is_some_and(|f| f != plan.short_name()) {
        return Vec::new();
    }
    let n = group.shared.len();
    let lanes = group.lanes.iter().map(|l| Requirement::Lane(l.stmt));
    let bounds = indices(group).into_iter().map(|i| in_bounds(i, n));
    let (pre, post): (Vec<Requirement>, Vec<Requirement>) = if plan.proves_bounds() {
        (Vec::new(), lanes.chain(bounds).collect())
    } else {
        (bounds.collect(), lanes.collect())
    };
    vec![Candidate { pre, post, cost: cost::read_group(plan, group, cx), emitter }]
}

// ── Select chain ──────────────────────────────────────────────────────

/// `x[i]` as a select chain: `acc_0 = x[0]`, `acc_k = [i == k] ? x[k] :
/// acc_{k-1}`, the read is `acc_{n-1}`.
///
/// **Preconditions.** `0 <= i < n` (the program's check).
/// **Postconditions.** The lanes.
/// **Soundness and completeness.** No hint. For `i` in `[0, n)` exactly the
/// test `[i == i]` holds among `k >= 1` when `i >= 1` (the last select
/// that fires is at `k = i`) and none when `i = 0` (`acc = x[0]`); so the
/// chain is `x[i]`. The equality tests are exact gadgets on every backend.
/// **Cost.** `n - 1` selects per read, `n - 1` equality tests per index.
pub struct ReadBySelectChain;

impl Construction for ReadBySelectChain {
    fn name(&self) -> &'static str {
        "memory: read by select chain"
    }
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        candidate(group, cx, ReadPlan::SelectChain, Box::new(ChainEmitter { cells: group.shared.clone() }))
    }
}

struct ChainEmitter {
    cells: Vec<StmtId>,
}

impl Emitter for ChainEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let index = cx.value(lane.own[0]);
        let dtype = dtype_of(&cells);
        let mut acc = cells[0].clone();
        for (k, cell) in cells.iter().enumerate().skip(1) {
            let hit = equals(b, &index, k);
            acc = select_typed(b, dtype, &hit, cell, &acc);
        }
        acc
    }
}

// ── One-hot ───────────────────────────────────────────────────────────

/// `x[i] = sum_k [i == k] x[k]` with `sum_k [i == k] = 1` asserted.
///
/// **Preconditions.** None.
/// **Postconditions.** The lanes; `0 <= i < n` for every index.
/// **Soundness and completeness.** No hint. The tests are bits; distinct
/// `k` cannot both equal `i`, so their sum is 1 exactly when `i` is one of
/// `0..n`, and then the inner product picks `x[i]`. An `i` outside has no
/// proof, which is what the dropped program check would have said.
/// **Cost.** `n` multiply-adds per read; `n` equality tests and the
/// exactly-one sum per index.
pub struct ReadByOneHot;

impl Construction for ReadByOneHot {
    fn name(&self) -> &'static str {
        "memory: read by one-hot"
    }
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        candidate(group, cx, ReadPlan::OneHot, Box::new(OneHotEmitter { cells: group.shared.clone() }))
    }
}

struct OneHotEmitter {
    cells: Vec<StmtId>,
}

impl Emitter for OneHotEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let index = cx.value(lane.own[0]);
        let dtype = dtype_of(&cells);
        let hits: Vec<Value> = (0..cells.len()).map(|k| equals(b, &index, k)).collect();
        let count = hits[1..].iter().fold(hits[0].clone(), |acc, h| b.ir_add_i(&acc, h));
        let one = b.ir_constant_int(1);
        let exactly_one = b.ir_equal_i(&count, &one);
        b.ir_assert(&exactly_one);
        let terms: Vec<Value> = hits.iter().zip(&cells).map(|(h, c)| scale_by_bit(b, h, c, dtype)).collect();
        sum_typed(b, terms, dtype)
    }
}

// ── Index bits (binary multiplexer) ───────────────────────────────────

/// `x[i]` by a binary multiplexer over `i`'s bits: `i < n` asserted with
/// `w = ceil(log2 n)` bits exposed (`AssertU128LtBit`), level `b` of the tree
/// selects between neighbouring entries of level `b - 1` by bit `b`; an
/// entry without a right neighbour passes up unchanged. For `n = 2` the
/// index is its own bit: one `select_bit` (which asserts the bit).
///
/// **Preconditions.** None.
/// **Postconditions.** The lanes; `0 <= i < n` for every index.
/// **Soundness and completeness.** No hint. The bits are the binary digits
/// of `i` and `i < n` is asserted (`assert_u128_lt_bits`: the backend
/// decomposes `i` into `w` booleans that recompose it and range-checks
/// `n - 1 - i`), so entry `i` of level 0 reaches the root: at each level
/// the entry holding `x[i]` sits at position `i >> b`, and a pass-through
/// entry is never the one a valid `i` needs a right neighbour for (its
/// neighbour's position would be `>= n`).
/// **Cost.** `n - 1` selects per read; the bits and the bound per index.
pub struct ReadByIndexBits;

impl Construction for ReadByIndexBits {
    fn name(&self) -> &'static str {
        "memory: read by index bits"
    }
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        candidate(group, cx, ReadPlan::IndexBits, Box::new(BitsEmitter { cells: group.shared.clone() }))
    }
}

struct BitsEmitter {
    cells: Vec<StmtId>,
}

impl Emitter for BitsEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let cells = cx.values(&self.cells);
        let index = cx.value(lane.own[0]);
        let dtype = dtype_of(&cells);
        if cells.len() == 2 {
            return match dtype {
                NumberType::Float => b.ir_select_bit_f(&index, &cells[1], &cells[0]),
                _ => b.ir_select_bit_i(&index, &cells[1], &cells[0]),
            };
        }
        let bits = b.ir_assert_u128_lt_bits(&index, cells.len());
        let mut level = cells;
        for bit in &bits {
            level = level
                .chunks(2)
                .map(|pair| match pair {
                    [lo, hi] => select_typed(b, dtype, bit, hi, lo),
                    [one] => one.clone(),
                    _ => unreachable!("chunks of two"),
                })
                .collect();
        }
        level.swap_remove(0)
    }
}

// ── Backend memory ────────────────────────────────────────────────────

/// The line written to a memory segment once (constant addresses: wiring),
/// every read a dynamic memory read (`IR::ReadMemory`).
///
/// **Preconditions.** `0 <= i < n` (the program's check: the mock's
/// memory read does not reject an address past the end, so this
/// construction does not claim the bound although both proving backends
/// enforce it).
/// **Postconditions.** The lanes.
/// **Soundness and completeness.** The backends' dynamic read proves the
/// address in bounds and returns the cell there: Noir reads a native array
/// at an index checked by `array_index` (`0 <= i <= n - 1`), or gathers by
/// a one-hot with a membership assertion; halo2-base gathers by a one-hot
/// with a membership assertion over the reachable slots, or by a
/// multiplexer after `i < n` (`synthesizer/memory.rs`). Each of these has no
/// witness for an out-of-range address and selects exactly the addressed
/// cell otherwise (their soundness arguments are at those functions).
/// **Cost.** Noir: the native array (3 gates per cell, once per line) and a
/// read plus a shared index check per read; halo2-base: the backend's
/// one-hot gather (equality tests per index, an inner product per read).
pub struct ReadByMemory;

impl Construction for ReadByMemory {
    fn name(&self) -> &'static str {
        "memory: read through a memory segment"
    }
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        candidate(group, cx, ReadPlan::Memory, Box::new(MemoryEmitter { cells: group.shared.clone(), segment: None }))
    }
}

struct MemoryEmitter {
    cells: Vec<StmtId>,
    segment: Option<(u32, NumberType)>,
}

impl Emitter for MemoryEmitter {
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let cells = cx.values(&self.cells);
        let dtype = dtype_of(&cells);
        self.segment = Some((crate::helpers::segment::alloc_and_write_values(b, &cells, dtype), dtype));
    }

    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let (segment, dtype) = self.segment.expect("begin ran");
        let index = cx.value(lane.own[0]);
        crate::helpers::segment::read_typed(b, segment, &index, dtype)
    }
}

// ── helpers ───────────────────────────────────────────────────────────

/// `[i == k]` for a constant `k` (duplicates across reads are merged by the
/// cleanup after lowering).
fn equals(b: &mut IRBuilder, index: &Value, k: usize) -> Value {
    let k = b.ir_constant_int(k as i64);
    b.ir_equal_i(index, &k)
}

fn select_typed(b: &mut IRBuilder, dtype: NumberType, cond: &Value, if_true: &Value, if_false: &Value) -> Value {
    match dtype {
        NumberType::Float => b.ir_select_f(cond, if_true, if_false),
        _ => b.ir_select_i(cond, if_true, if_false),
    }
}
