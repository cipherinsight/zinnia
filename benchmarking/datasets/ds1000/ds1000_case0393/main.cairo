//! DS-1000 case 0393: the columns `low` to `high - 1` of a matrix,
//! `a[:, low:high]`.
//!
//!     def kernel(a: Float[Array, "20 10"]):
//!         low = 4
//!         high = 7
//!         high = min(high, a.shape[1])
//!         result = a[:, low:high]
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 20;
const COLS: u32 = 10;
const LOW: u32 = 4;
/// `min(high, a.shape[1]) = min(7, COLS)`.
const HIGH: u32 = 7;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    for r in 0..ROWS {
        result.append_span(a.slice(r * COLS + LOW, HIGH - LOW));
    }
    result
}
