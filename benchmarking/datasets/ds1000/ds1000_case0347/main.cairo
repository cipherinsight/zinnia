//! DS-1000 case 0347: divide row `row - 1` of `a` and return the product of
//! its entries.
//!
//!     def kernel(a: Float[Array, "8 6"], row: Integer, divide_number: Float):
//!         a[row-1, :] /= divide_number
//!         result = np.multiply.reduce(a[row-1, :])
//!         return result
// output: q32
use crate::np::np_index;
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>, row: i64, divide_number: Q32) -> Q32 {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let selected = a.span().slice(np_index(row - 1, ROWS) * COLS, COLS);
    let mut product = *selected.at(0) / divide_number;
    for j in 1..COLS {
        product = product * (*selected.at(j) / divide_number);
    }
    product
}
