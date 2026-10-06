//! NPBench polybench/cholesky2: the Cholesky factor `L` of `A` in the lower
//! triangle, the strict upper triangle kept, in place.
//!
//! ```python
//! def kernel(A: Float[Array, '1000 1000']):
//!     A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
//! ```
//!
//! `np.linalg.cholesky` reads the lower triangle of `A` and raises unless it
//! is positive definite. Each entry of `L` is an inner product of entries
//! already computed (LAPACK `potrf`), fused here like a library dot product.
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated A
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut a: Array2<Q32> = args.array2();
        let n = a.nrows();
        assert_eq!(a.ncols(), n, "Last 2 dimensions of the array must be square");
        // Row by row, L[i, j] overwrites A[i, j] (j <= i) once nothing reads
        // the original any more; the strict upper triangle is never written.
        for i in 0..n {
            for j in 0..=i {
                let s = dot((0..j).map(|k| (a[(i, k)], a[(j, k)])));
                let r = a[(i, j)] - s;
                a[(i, j)] = if i == j {
                    assert!(r > Q32::ZERO, "Matrix is not positive definite");
                    r.sqrt()
                } else {
                    r / a[(j, j)]
                };
            }
        }
        a
    }
}
