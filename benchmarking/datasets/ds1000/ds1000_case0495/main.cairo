//! DS-1000 case 0495: a matrix padded with zeros at the bottom and right to
//! `shape = (93, 13)`.
//!
//!     def kernel(a: Float[Array, "41 13"]):
//!         shape = (93, 13)
//!         result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant')
//!         return result
// output: [q32]
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 41;
/// Also the target width (`shape[1]`), so only zero rows are added.
const COLS: u32 = 13;
const OUT_ROWS: u32 = 93;

/// The rows of `a` followed by `OUT_ROWS - ROWS` rows of zeros.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut result = a;
    for _ in 0..(OUT_ROWS - ROWS) * COLS {
        result.append(ZERO);
    }
    result
}
