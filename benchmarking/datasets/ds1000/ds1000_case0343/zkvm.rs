//! DS-1000 case 0343: the elements of `X` in column-major order (`X.T.flat`).
//!
//! ```python
//! def kernel(X: Integer[Array, "5 6"]):
//!     result = []
//!     for value in X.T.flat:
//!         result.append(value)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 6;

pub type Output = Vec<i64>;

/// Row `j` of `X.T` is column `j` of `X`.
pub fn main(input: &mut impl Input) -> Output {
    let x: Vec<i64> = input.read_vec(ROWS * COLS);
    let mut result = Vec::with_capacity(ROWS * COLS);
    for j in 0..COLS {
        for i in 0..ROWS {
            result.push(x[i * COLS + j]);
        }
    }
    result
}
