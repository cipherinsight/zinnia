//! DS-1000 case 0435: zero row 1 and column 0 of a 4x4 matrix.
//!
//! ```python
//! def kernel(a: Integer[Array, "4 4"]):
//!     a[1, :] = 0
//!     a[:, 0] = 0
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 4;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let mut a: [[i64; N]; N] = input.read_matrix();
    a[1] = [0; N];
    for row in &mut a {
        row[0] = 0;
    }
    a.concat()
}
