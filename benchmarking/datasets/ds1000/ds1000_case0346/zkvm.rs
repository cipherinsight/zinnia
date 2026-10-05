//! DS-1000 case 0346: scale row `row - 1` of `a` and return its running sum.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"], row: Integer, multiply_number: Float):
//!     a[row-1, :] *= multiply_number
//!     result = np.cumsum(a[row-1, :])
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; COLS]; ROWS] = input.read_matrix();
    let row: i64 = input.read();
    let multiply_number: Q32 = input.read();
    let mut sum = Q32::ZERO;
    a[np_index(row - 1, ROWS)]
        .iter()
        .map(|&v| {
            sum += v * multiply_number;
            sum
        })
        .collect()
}
