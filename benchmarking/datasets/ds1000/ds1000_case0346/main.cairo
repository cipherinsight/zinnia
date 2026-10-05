//! DS-1000 case 0346: scale row `row - 1` of `a` and return its running sum.
//!
//!     def kernel(a: Float[Array, "8 6"], row: Integer, multiply_number: Float):
//!         a[row-1, :] *= multiply_number
//!         result = np.cumsum(a[row-1, :])
//!         return result
// output: [q32]
use crate::np::np_index;
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>, row: i64, multiply_number: Q32) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let selected = a.span().slice(np_index(row - 1, ROWS) * COLS, COLS);
    let mut sum = ZERO;
    let mut result = array![];
    for v in selected {
        sum = sum + *v * multiply_number;
        result.append(sum);
    }
    result
}
