//! Element gathers: `x[i]` over `n` cells at a runtime index `i`.
//!
//! Three generators, all for an `i` that lies in `[0, n)` wherever the read
//! runs (the frontend wrapped NumPy's negative index and checked it, or `i`
//! is a scan's result):
//!
//! ```text
//! by a recorded one-hot:   x[i] = sum_k onehot[k] * x[k]
//! by a select chain:       x[i] = select([i == n-1], x[n-1], ... select([i == 1], x[1], x[0]))
//! by indicators:           x[i] = sum_k [i == k] * x[k]
//! ```
//!
//! **Preconditions.** `0 <= i < n` where the result is used (the frontend's
//! index check, which a scan's range usually discharges); for the first
//! form, `onehot` is the encoding of `i`.
//!
//! **Postcondition.** The result is `x[i]`.
//!
//! **Soundness.** No hint. Every bit is exact (an equality test, or a bit of
//! a recorded encoding), and for an `i` in range exactly one of `[i == 0],
//! ..., [i == n-1]` is 1: the chain then passes every select whose test is 0
//! and stops at the one that is 1 (or reaches `x[0]` when `i == 0`), and
//! the sum has exactly one non-zero term, `x[i]`. Where the read does not
//! run, its value is not used (it is guarded), so the result for an index
//! out of range (the chain gives `x[0]`, the sums 0) does not matter.
//!
//! **Cost** (`cost.rs`). The equality tests of one index are shared by every
//! read at that index (the lowering deduplicates them). Per read, the chain
//! has `n - 1` selects of two variable branches, the inner products `n`
//! multiply-adds. halo2-base: a select chain over the equality bits of a
//! bounded index is already an inner product in the backend
//! (`prove/halo2_base/synthesizer/plan.rs`, `OneHotChain`), so the chain is
//! never dearer there and needs one equality test less. Noir: a select of
//! two variable branches is two gates (`c * a + (1 - c) * b`), a
//! multiply-add one, so the inner product wins once the index's `[i == 0]`
//! is shared by a few reads.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

use super::encoding::IndexEncoding;
use super::window::{scale_by_bit, sum_typed};

/// `cells[i]` as the inner product of the cells with `encoding`, the
/// one-hot of `i`.
pub fn gather_by_onehot(b: &mut IRBuilder, cells: &[Value], encoding: &IndexEncoding, dtype: NumberType) -> Value {
    let terms: Vec<Value> = cells
        .iter()
        .enumerate()
        .filter_map(|(k, cell)| {
            let bit = encoding.onehot.get(usize::try_from(k as i64 - encoding.lo).ok()?)?.clone();
            Some(scale_by_bit(b, &bit, cell, dtype))
        })
        .collect();
    if terms.is_empty() {
        return zero(b, dtype);
    }
    sum_typed(b, terms, dtype)
}

/// `cells[i]` (not empty) as a select chain over the equality bits of `i`:
/// from `cells[0]`, select `cells[k]` under `[i == k]` for `k = 1..n`.
pub fn gather_by_select_chain(b: &mut IRBuilder, cells: &[Value], index: &Value, dtype: NumberType) -> Value {
    let (first, rest) = cells.split_first().expect("an element read over no cells");
    rest.iter().enumerate().fold(first.clone(), |acc, (k, cell)| {
        let hit = equals(b, index, k as i64 + 1);
        match dtype {
            NumberType::Float => b.ir_select_f(&hit, cell, &acc),
            _ => b.ir_select_i(&hit, cell, &acc),
        }
    })
}

/// `cells[i]` (not empty) as the inner product of the cells with the
/// equality bits `[i == k]`.
pub fn gather_by_indicators(b: &mut IRBuilder, cells: &[Value], index: &Value, dtype: NumberType) -> Value {
    let terms: Vec<Value> = cells
        .iter()
        .enumerate()
        .map(|(k, cell)| {
            let hit = equals(b, index, k as i64);
            scale_by_bit(b, &hit, cell, dtype)
        })
        .collect();
    sum_typed(b, terms, dtype)
}

/// `[index == k]`.
fn equals(b: &mut IRBuilder, index: &Value, k: i64) -> Value {
    let k = b.ir_constant_int(k);
    b.ir_equal_i(index, &k)
}

fn zero(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => b.ir_constant_float(0.0),
        _ => b.ir_constant_int(0),
    }
}
