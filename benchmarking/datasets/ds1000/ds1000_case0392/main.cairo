//! DS-1000 case 0392: the rows `low` to `high - 1` of a matrix,
//! `a[low:high, :]`.
//!
//!     def kernel(a: Float[Array, "20 10"]):
//!         low = 4
//!         high = 7
//!         result = a[low:high, :]
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 20;
const COLS: u32 = 10;
const LOW: u32 = 4;
const HIGH: u32 = 7;

/// Whole rows are contiguous in row-major order.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut result = array![];
    result.append_span(a.span().slice(LOW * COLS, (HIGH - LOW) * COLS));
    result
}
