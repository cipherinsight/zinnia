//! The memory family's constructions for element reads (`OpKind::Element`
//! groups: reads of one line of `n` cells, each at a runtime index `i` that
//! the frontend checked into `[0, n)` where the read runs).

use std::any::Any;
use std::rc::Rc;

use crate::builder::IRBuilder;
use crate::lower::{
    Bound, Candidate, Claim, Construction, Context, EmitCx, Emitter, Fact, Group, Lane, OpKind, Requirement, Scope,
};
use crate::types::{NumberType, StmtId, Value};

use super::cost;

/// The family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&ElementByIndexBits, &IndexBitsNode]
}

/// `0 <= i < n` on every path. The index operand is the frontend's guarded
/// index (`discharge_index_in_range` returns `select(p, i, 0)` under an
/// execution predicate `p`), which lies in range where the read does not run.
fn in_bounds(index: StmtId, n: usize) -> Requirement {
    Requirement::Fact(Fact { claim: Claim::InRange { x: index, lo: 0, hi: Bound::Const(n as i64) }, scope: Scope::Everywhere })
}

/// `x[i]` by a binary multiplexer over `i`'s bits.
///
/// For `n >= 3`: `i < n` is asserted with its `w = ceil(log2 n)` low bits
/// exposed (`IR::AssertU128LtBit`), once per index (a separate node, see
/// [`IndexBitsNode`]); level `b` of the tree selects between neighbouring
/// entries of level `b - 1` by bit `b`, and an entry without a right
/// neighbour passes up unchanged. For `n = 2` the index is its own bit: one
/// `select_bit` per read (which asserts the bit).
///
/// **Preconditions.** `n >= 3`: the bits of every index it reads (the
/// shared value of [`IndexBitsNode`], which proves `0 <= i < n`). None for
/// `n = 2`.
///
/// **Postconditions.** Every lane; for `n = 2`, `0 <= i < 2` for every
/// index read.
///
/// **Soundness and completeness.** No hint. The bits are `i`'s binary
/// digits and `0 <= i < n` (the bits node's postcondition). For `i < n` the
/// entry holding `x[i]` is at position `i >> b` after level `b`; the
/// pass-through entry of an odd-length level never needs its missing right
/// neighbour, because that neighbour's positions start at `n` or later. So
/// the root is `x[i]`. For `n = 2`, `select_bit` asserts `i (i - 1) = 0`, so
/// `i` is 0 or 1 (an index outside has no witness, as NumPy raises), and
/// selects `x[i]`.
///
/// **Cost** (`cost::index_bits_read`): `n - 1` selects of two variable
/// branches per read (`n = 2`: one `select_bit`).
pub struct ElementByIndexBits;

/// The bound `0 <= i < n` and `i`'s `w = ceil(log2 n)` bits
/// (`ir_assert_u128_lt_bits`) for one index `i`, shared by every multiplexer
/// that reads at `i` over a line of `n >= 3` cells (a
/// `Requirement::SharedAt` of `i`). Offered by the first group that reads
/// at `i` over `n` cells.
///
/// **Preconditions.** None.
///
/// **Postconditions.** The bits; `0 <= i < n`.
///
/// **Soundness and completeness.** `AssertU128LtBit` constrains `w`
/// booleans that recompose `i` and range-checks `n - 1 - i` on every backend
/// (the mock rejects `i` outside `[0, n)` too), so the bits are `i`'s binary
/// digits exactly when `0 <= i < n`, and there is no witness otherwise
/// (NumPy raises, and the program's check, which this postcondition
/// replaces, had none either).
///
/// **Cost** (`cost::index_bits`): the range check and the bits.
pub struct IndexBitsNode;

/// The name of [`IndexBitsNode`]'s shared value (one per index statement).
const INDEX_BITS: &str = "memory: index bits";

/// The length of the lines read at `index` by element groups, when they all
/// have the same length (the bits node's bound); `None` otherwise (then no
/// multiplexer reads at `index` over a line of 3 or more cells).
fn line_length_at(cx: &Context, index: StmtId) -> Option<usize> {
    let mut lengths = cx
        .groups
        .iter()
        .filter(|g| g.kind == OpKind::Element && g.lanes.iter().any(|l| l.own[0] == index))
        .map(|g| g.shared.len());
    let first = lengths.next()?;
    lengths.all(|n| n == first).then_some(first)
}

/// Whether `group` is the first element group that reads at `index`.
fn first_reader(cx: &Context, group: &Group, index: StmtId) -> bool {
    cx.groups.iter().find(|g| g.kind == OpKind::Element && g.lanes.iter().any(|l| l.own[0] == index)).map(|g| g.id)
        == Some(group.id)
}

fn distinct_indices(group: &Group) -> Vec<StmtId> {
    let mut indices: Vec<StmtId> = group.lanes.iter().map(|l| l.own[0]).collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

impl Construction for ElementByIndexBits {
    fn name(&self) -> &'static str {
        "memory: element by index bits"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let n = group.shared.len();
        if group.kind != OpKind::Element || n < 2 {
            return Vec::new();
        }
        let indices = distinct_indices(group);
        let lanes = group.lanes.iter().map(|l| Requirement::Lane(l.stmt));
        let cost = group.lanes.len() as u64 * cost::index_bits_read(n as u64, cx.target);
        let emitter = Box::new(BitsEmitter { cells: group.shared.clone() });
        if n == 2 {
            let post = lanes.chain(indices.into_iter().map(|i| in_bounds(i, n)));
            return vec![Candidate { pre: Vec::new(), post: post.collect(), cost, emitter }];
        }
        if indices.iter().any(|&i| line_length_at(cx, i) != Some(n)) {
            return Vec::new();
        }
        let pre = indices.into_iter().map(|at| Requirement::SharedAt { at, name: INDEX_BITS }).collect();
        vec![Candidate { pre, post: lanes.collect(), cost, emitter }]
    }
}

impl Construction for IndexBitsNode {
    fn name(&self) -> &'static str {
        "memory: index bits"
    }

    /// One node per index this group is the first to read (over 3 or more
    /// cells, every reader of the index over the same length).
    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        let n = group.shared.len();
        if group.kind != OpKind::Element || n < 3 {
            return Vec::new();
        }
        distinct_indices(group)
            .into_iter()
            .filter(|&i| first_reader(cx, group, i) && line_length_at(cx, i) == Some(n))
            .map(|at| Candidate {
                pre: Vec::new(),
                post: vec![Requirement::SharedAt { at, name: INDEX_BITS }, in_bounds(at, n)],
                cost: cost::index_bits(n as u64, cx.target),
                emitter: Box::new(BitsNodeEmitter { n }),
            })
            .collect()
    }
}

struct BitsNodeEmitter {
    n: usize,
}

impl Emitter for BitsNodeEmitter {
    fn lane(&mut self, _b: &mut IRBuilder, _cx: &mut EmitCx, _lane: &Lane) -> Value {
        unreachable!("the index bits node produces no lane")
    }

    fn shared_at(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, at: StmtId, _name: &'static str) -> Rc<dyn Any> {
        Rc::new(b.ir_assert_u128_lt_bits(&cx.value(at), self.n))
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
        let bits: Rc<Vec<Value>> =
            if cells.len() == 2 { Rc::new(Vec::new()) } else { cx.shared_at(b, lane.own[0], INDEX_BITS) };
        read_by_index_bits(b, &cells, &index, &bits, dtype)
    }
}

/// `cells[index]` (`cells.len() >= 2`) by a multiplexer over `bits`, the
/// index's binary digits (least significant first; empty for two cells,
/// where the index is its own bit and `select_bit` asserts it). See
/// [`ElementByIndexBits`].
pub fn read_by_index_bits(b: &mut IRBuilder, cells: &[Value], index: &Value, bits: &[Value], dtype: NumberType) -> Value {
    if let [lo, hi] = cells {
        return match dtype {
            NumberType::Float => b.ir_select_bit_f(index, hi, lo),
            _ => b.ir_select_bit_i(index, hi, lo),
        };
    }
    let mut level = cells.to_vec();
    for bit in bits {
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [lo, hi] => match dtype {
                    NumberType::Float => b.ir_select_f(bit, hi, lo),
                    _ => b.ir_select_i(bit, hi, lo),
                },
                [one] => one.clone(),
                _ => unreachable!("chunks of two"),
            })
            .collect();
    }
    assert_eq!(level.len(), 1, "the bits cover the cells");
    level.swap_remove(0)
}

fn dtype_of(cells: &[Value]) -> NumberType {
    if cells.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}
