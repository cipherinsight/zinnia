//! DS-1000 case 0345: scale column `col - 1` of `a` and return its running
//! sum.
//!
//!     def kernel(a: Float[Array, "8 6"], col: Integer, multiply_number: Float):
//!         a[:, col-1] *= multiply_number
//!         result = np.cumsum(a[:, col-1])
//!         return result
// output: [q32]
use crate::np::np_index;
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>, col: i64, multiply_number: Q32) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let j = np_index(col - 1, COLS);
    let mut sum = ZERO;
    let mut result = array![];
    for r in 0..ROWS {
        sum = sum + *a.at(r * COLS + j) * multiply_number;
        result.append(sum);
    }
    result
}
