//! NPBench polybench/seidel_2d: Gauss-Seidel sweeps of a 9-point stencil, in
//! place.
//!
//! ```python
//! @requires(lambda TSTEPS, N, A: 0 <= N <= 1024)
//! def kernel(TSTEPS: Integer, N: Integer, A: Float[Array, '50 50']):
//!     for t in range(0, TSTEPS - 1):
//!         for i in range(1, N - 1):
//!             A[i, 1:-1] += A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] + A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] + A[i + 1, 2:]
//!             for j in range(1, N - 1):
//!                 A[i, j] += A[i, j - 1]
//!                 A[i, j] /= 9.0
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated A
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let tsteps: i64 = args.scalar();
        let n: i64 = args.scalar();
        let mut a: Array2<Q32> = args.array2();
        assert!((0..=1024).contains(&n), "requires 0 <= N <= 1024");
        let n = n as usize;
        let cols = a.ncols();
        for _ in 0..tsteps - 1 {
            for i in 1..n.saturating_sub(1) {
                // The slice update over the columns 1..cols-1 reads A[i, j + 1]
                // before it changes, as NumPy's right-hand side does, so it can
                // run left to right in place.
                for j in 1..cols.saturating_sub(1) {
                    let neighbours = a[(i - 1, j - 1)]
                        + a[(i - 1, j)]
                        + a[(i - 1, j + 1)]
                        + a[(i, j + 1)]
                        + a[(i + 1, j - 1)]
                        + a[(i + 1, j)]
                        + a[(i + 1, j + 1)];
                    a[(i, j)] += neighbours;
                }
                for j in 1..n - 1 {
                    a[(i, j)] = (a[(i, j)] + a[(i, j - 1)]) / 9;
                }
            }
        }
        a
    }
}
