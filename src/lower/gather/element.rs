//! Element gather by an index whose one-hot is already known.
//!
//! `x[i]` at a runtime `i` is lowered by the backends as a memory read:
//! halo2-base builds the indicators `[i == k]` (an equality test each,
//! shared by the reads of the same index) and an inner product; Noir uses a
//! native array read or the same one-hot gather. When `i` is a scan's result
//! (plus a constant), the scan's running products are its one-hot already
//! (`scan.rs`), and the read is the inner product alone:
//!
//! ```text
//! x[i] = sum_k [i == k] * x[k]
//! ```
//!
//! **Preconditions.** `encoding` is the encoding of `i`; `0 <= i < n` (the
//! frontend's index check, which the scan's range usually discharges).
//!
//! **Postcondition.** The result is `x[i]`.
//!
//! **Soundness.** No hint: the encoding's bits are exact and exactly one is
//! 1 for an `i` in range, so the sum is `x[i]`.
//!
//! **Cost** (`cost.rs`): halo2-base 3 cells per element (a multiply-add
//! chain), against an equality test (7 cells) per element plus the same
//! chain for a fresh one-hot; Noir one gate per element, against about 10
//! gates plus 3 per element shared by the reads of a native array.

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
        return match dtype {
            NumberType::Float => b.ir_constant_float(0.0),
            _ => b.ir_constant_int(0),
        };
    }
    sum_typed(b, terms, dtype)
}
