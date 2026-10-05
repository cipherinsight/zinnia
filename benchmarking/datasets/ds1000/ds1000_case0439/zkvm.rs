//! DS-1000 case 0439: the outer product of every column of `X` with itself,
//! `X.T[:, :, None] * X.T[:, None]`, of shape `(20, 10, 10)`:
//! `result[i][j][k] = X[j][i] * X[k][i]`.
//!
//! ```python
//! def kernel(X: Float[Array, "10 20"]):
//!     result = X.T[:, :, None] * X.T[:, None]
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 10;
const COLS: usize = 20;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let x: [[Q32; COLS]; ROWS] = input.read_matrix();
    let mut out = Vec::with_capacity(COLS * ROWS * ROWS);
    for i in 0..COLS {
        for row_j in &x {
            for row_k in &x {
                out.push(row_j[i] * row_k[i]);
            }
        }
    }
    out
}
