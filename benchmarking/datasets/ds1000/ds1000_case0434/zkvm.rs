//! DS-1000 case 0434: zero the listed rows and columns of a 4x4 matrix.
//!
//! ```python
//! def kernel(a: Integer[Array, "4 4"], zero_rows: Integer[Array, "2"], zero_cols: Integer[Array, "2"]):
//!     a[zero_rows, :] = 0
//!     a[:, zero_cols] = 0
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 4;
const K: usize = 2;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let mut a: [[i64; N]; N] = input.read_matrix();
    let zero_rows: [i64; K] = input.read_array();
    let zero_cols: [i64; K] = input.read_array();
    for r in zero_rows {
        a[np_index(r, N)] = [0; N];
    }
    for c in zero_cols {
        let c = np_index(c, N);
        for row in &mut a {
            row[c] = 0;
        }
    }
    a.concat()
}
