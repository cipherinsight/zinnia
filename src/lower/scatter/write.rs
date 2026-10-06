//! A write at a runtime index on one axis: `a[i, ...] = v` (the other axes
//! static), `OpKind::IndexWrite`.
//!
//! Terms. `n` is the length of the indexed axis; `P` is NumPy's position of
//! `i` (`i`, or `i + n` for a negative `i`). Every cell the write may touch
//! is one *lane*: its old value, the value written to it, optionally the
//! execution predicate `p` of the write, and its coordinate `c` on the
//! indexed axis (the lane's position). Its new value is `v` when `P = c`
//! (and `p` holds), else the old value.
//!
//! # Construction: one-hot write ([`OneHotWrite`])
//!
//! The pinned one-hot `h[c] = [P == c]` of `super::position` (element
//! positions), shared by every lane of the group (every write through the
//! same index on an axis of the same length), and per lane
//! `select(h[c] and p, v, old)`.
//!
//! **Preconditions**: `n` is a compile-time constant; the index operand is
//! guarded by the execution predicate (0 where the write does not run), so
//! the one-hot's checks hold on every path.
//! **Postconditions**: the new cells; the fact `-n <= i < n` (no witness
//! otherwise, as NumPy raises `IndexError`).
//! **Soundness**: `position_one_hot`'s argument pins `h`; each lane is then
//! a select on a bit, a function of its inputs.
//! **Cost**: the one-hot (halo2-base about 10 cells per position plus 15;
//! Noir about 2 gates per position plus 5) and a select per lane
//! (halo2-base 4 cells writing 0, else 8; Noir 1-2 gates).
//!
//! # Construction: zero-write chain ([`ZeroWrites`])
//!
//! Writes of the constant 0 through `k` indices on axes of one length, one
//! after the other over the same cells (`a[rows, :] = 0` with an index
//! array writes one index at a time): the cell at coordinate `c` ends as
//! `x * keep[c]` with `keep[c] = prod_t (1 - h_t[c])`, the product shared
//! by every cell at that coordinate. One product per cell instead of `k`
//! selects; `k - 1` products per coordinate.
//!
//! **Preconditions**: as the one-hot write, for each index; the
//! intermediate cells are read only by the next write.
//! **Postconditions**: every write's cells (the intermediate ones are
//! computed by selects, then unused); `-n <= i_t < n` for each index.
//! **Soundness**: each `h_t` is pinned; `x * prod_t (1 - h_t[c])` is 0
//! exactly when some index hits `c`, NumPy's result of the zero writes.
//! **Cost**: the `k` one-hots, `k - 1` products per coordinate, one per
//! cell (halo2-base 4 cells per product; Noir 1 gate).

use crate::builder::IRBuilder;
use crate::lower::Target;
use crate::types::{NumberType, Value};

use super::position::{position_one_hot, PositionOneHot, Positions};

/// The operands of one lane: `[old, written]`, then the execution predicate
/// when the write is conditional.
pub const OLD: usize = 0;
pub const WRITTEN: usize = 1;
pub const PREDICATE: usize = 2;

/// Cost of the one-hot write of `lanes` cells through an index on an axis
/// of length `n`, `zeros` of them writing the constant 0.
pub fn one_hot_cost(target: Target, n: usize, lanes: usize, zeros: usize) -> u64 {
    let (lanes, zeros) = (lanes as u64, zeros as u64);
    index_cost(target, n)
        + match target {
            Target::Halo2Base => 4 * zeros + 8 * (lanes - zeros),
            Target::Noir => 2 * lanes,
        }
}

/// Cost of the pinned one-hot of one index on an axis of length `n`.
fn index_cost(target: Target, n: usize) -> u64 {
    let n = n as u64;
    match target {
        Target::Halo2Base => 10 * n + 15,
        Target::Noir => 2 * n + 5,
    }
}

/// Cost of the zero-write chain of `writes` indices on an axis of length
/// `n`, ending in `lanes` cells.
pub fn zero_chain_cost(target: Target, n: usize, writes: usize, lanes: usize) -> u64 {
    let products = (n * (writes - 1) + lanes) as u64;
    writes as u64 * index_cost(target, n)
        + match target {
            Target::Halo2Base => 4 * products,
            Target::Noir => products,
        }
}

/// The one-hot of the index of a group of writes.
pub struct WriteIndex {
    point: PositionOneHot,
}

impl WriteIndex {
    pub fn new(b: &mut IRBuilder, index: &Value, n: usize) -> WriteIndex {
        WriteIndex { point: position_one_hot(b, index, n, Positions::Element) }
    }

    /// The new value of a cell at coordinate `coord` on the indexed axis.
    pub fn cell(&self, b: &mut IRBuilder, coord: usize, own: &[Value]) -> Value {
        let hit = &self.point.hot[coord];
        let cond = match own.get(PREDICATE) {
            Some(p) => b.ir_logical_and(hit, p),
            None => hit.clone(),
        };
        let (written, old) = (&own[WRITTEN], &own[OLD]);
        if matches!(written, Value::Float(_)) || matches!(old, Value::Float(_)) {
            b.ir_select_f(&cond, written, old)
        } else {
            b.ir_select_i(&cond, written, old)
        }
    }
}

/// The one-hots of a chain of zero writes and their shared keep products.
pub struct ZeroWrites {
    writes: Vec<WriteIndex>,
    /// `keep[c] = prod_t (1 - h_t[c])` over every write of the chain.
    keep: Vec<Value>,
}

impl ZeroWrites {
    pub fn new(b: &mut IRBuilder, indices: &[Value], n: usize) -> ZeroWrites {
        let writes: Vec<WriteIndex> = indices.iter().map(|i| WriteIndex::new(b, i, n)).collect();
        let keep = (0..n)
            .map(|c| {
                // (1 - h_1) * (1 - h_2) * ...: each further factor as a
                // select that zeroes the product on a hit.
                let zero = b.ir_constant_int(0);
                let mut keep = b.ir_logical_not(&writes[0].point.hot[c]);
                for write in &writes[1..] {
                    keep = b.ir_select_i(&write.point.hot[c], &zero, &keep);
                }
                keep
            })
            .collect();
        ZeroWrites { writes, keep }
    }

    /// The cell at `coord` after write `t` of the chain (`own` are that
    /// lane's operands).
    pub fn cell(&self, b: &mut IRBuilder, t: usize, coord: usize, own: &[Value], base: &Value) -> Value {
        if t + 1 < self.writes.len() {
            return self.writes[t].cell(b, coord, own);
        }
        let dtype = if matches!(base, Value::Float(_)) { NumberType::Float } else { NumberType::Integer };
        super::arith::bit_times(b, &self.keep[coord], base, dtype)
    }
}
