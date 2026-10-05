//! DS-1000 case 0337: the diagonal of `a` flipped left to right,
//! `np.diag(np.fliplr(a))`, for a wide matrix.
//!
//!     def kernel(a: Float[Array, "8 10"]):
//!         result = np.diag(np.fliplr(a))
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 10;

/// `np.fliplr` reverses every row, so entry `i` of its diagonal is
/// `a[i][COLS - 1 - i]`; the diagonal has `min(ROWS, COLS) = ROWS` entries.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut result = array![];
    for i in 0..ROWS {
        result.append(*a.at(i * COLS + (COLS - 1 - i)));
    }
    result
}
