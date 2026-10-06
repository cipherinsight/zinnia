//! Element gather by an index whose one-hot is already known.
//!
//! `x[i]` at a runtime `i` is lowered by the backends as a memory read:
//! halo2-base builds the indicators `[i == k]` (an equality test each,
//! shared by the reads of the same index) and an inner product; Noir uses a
//! native array read or the same one-hot gather. When `i` was produced by a
//! construction that records its one-hot (`IndexEncoding`, e.g. the argmax
//! of a flag vector, `scan.rs`), the indicators exist already and the read
//! is the inner product alone:
//!
//! ```text
//! x[i] = sum_k [i == k] * x[k]
//! ```
//!
//! **Preconditions.** `i` carries an encoding; `0 <= i < n` (the caller's
//! index check, which the encoding's range fact usually discharges).
//!
//! **Postcondition.** The result is `x[i]`.
//!
//! **Soundness.** No hint: the encoding's bits are exact, exactly one is 1
//! for an `i` in range, so the sum is `x[i]`.
//!
//! **Cost.** halo2-base 3 cells per element (multiply-add chain), against
//! one equality test (7-11 cells) per element plus the same chain for a
//! fresh one-hot; Noir one gate per element, against about 10 gates plus 3
//! per element shared by the reads of a native array (cheaper here unless
//! the array is long and read many times).

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

use super::encoding::encoding_of;
use super::window::{scale_by_bit, sum_typed};

/// `cells[index]` as an inner product with the index's recorded one-hot;
/// `None` when the index has no encoding or the array is empty.
pub fn gather_by_encoding(b: &mut IRBuilder, cells: &[Value], index: &Value, dtype: NumberType) -> Option<Value> {
    if cells.is_empty() {
        return None;
    }
    let enc = encoding_of(b, index)?;
    let terms: Vec<Value> = cells
        .iter()
        .enumerate()
        .filter_map(|(k, cell)| {
            let k = k as i64 - enc.lo;
            let bit = enc.onehot.get(usize::try_from(k).ok()?)?.clone();
            Some(scale_by_bit(b, &bit, cell, dtype))
        })
        .collect();
    if terms.is_empty() {
        return Some(match dtype {
            NumberType::Float => b.ir_constant_float(0.0),
            _ => b.ir_constant_int(0),
        });
    }
    Some(sum_typed(b, terms, dtype))
}
