//! DS-1000 case 0347: divide row `row - 1` of `a` and return the product of
//! its entries.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"], row: Integer, divide_number: Float):
//!     a[row-1, :] /= divide_number
//!     result = np.multiply.reduce(a[row-1, :])
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; COLS]; ROWS] = input.read_matrix();
    let row: i64 = input.read();
    let divide_number: Q32 = input.read();
    a[np_index(row - 1, ROWS)]
        .iter()
        .map(|&v| v / divide_number)
        .reduce(|product, v| product * v)
        .expect("rows are not empty")
}
