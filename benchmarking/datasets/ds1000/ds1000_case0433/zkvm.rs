//! DS-1000 case 0433: zero one row and one column of a 4x4 matrix.
//!
//! ```python
//! def kernel(a: Integer[Array, "4 4"], zero_rows: Integer, zero_cols: Integer):
//!     a[zero_rows, :] = 0
//!     a[:, zero_cols] = 0
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 4;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let mut a: [[i64; N]; N] = input.read_matrix();
    let zero_row = np_index(input.read(), N);
    let zero_col = np_index(input.read(), N);
    a[zero_row] = [0; N];
    for row in &mut a {
        row[zero_col] = 0;
    }
    a.concat()
}
