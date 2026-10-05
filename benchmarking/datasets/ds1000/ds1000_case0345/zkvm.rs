//! DS-1000 case 0345: scale column `col - 1` of `a` and return its running
//! sum.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"], col: Integer, multiply_number: Float):
//!     a[:, col-1] *= multiply_number
//!     result = np.cumsum(a[:, col-1])
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; COLS]; ROWS] = input.read_matrix();
    let col: i64 = input.read();
    let multiply_number: Q32 = input.read();
    let j = np_index(col - 1, COLS);
    let mut sum = Q32::ZERO;
    a.iter()
        .map(|row| {
            sum += row[j] * multiply_number;
            sum
        })
        .collect()
}
