//! Pythran matmul: matrix product with explicit loops over nested lists.
//!
//! ```python
//! def zero(n, m):
//!     return [[0 for row in range(n)] for col in range(m)]
//!
//! def matrix_multiply(m0: Float[Array, '60 60'], m1: Float[Array, '60 60']):
//!     new_matrix = zero(len(m0), len(m1[0]))
//!     for i in range(len(m0)):
//!         for j in range(len(m1[0])):
//!             r = 0
//!             for k in range(len(m1)):
//!                 r += m0[i][k] * m1[k][j]
//!             new_matrix[i][j] = r
//!     return new_matrix
//! ```
//!
//! `zero(n, m)` has `m` rows of `n` entries, so `new_matrix` is
//! `len(m1[0]) x len(m0)`: indexing it panics, as Python raises, unless
//! `len(m0) == len(m1[0])`.
use crate::prelude::*;

pub mod matrix_multiply {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let m0: Array2<Q32> = args.array2();
        let m1: Array2<Q32> = args.array2();
        assert!(m1.nrows() > 0, "m1[0] of an empty list");
        let mut product = Array2::zeros((m1.ncols(), m0.nrows()));
        for i in 0..m0.nrows() {
            for j in 0..m1.ncols() {
                // `r += m0[i][k] * m1[k][j]`: every product is rounded on its own.
                product[(i, j)] = (0..m1.nrows()).map(|k| m0[(i, k)] * m1[(k, j)]).sum();
            }
        }
        product
    }
}
