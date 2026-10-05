//! NPBench polybench/cholesky: Cholesky factorization, the lower triangle
//! of `A` overwritten by `L` with `A = L @ L.T` (the strict upper triangle
//! is left as it was), in place.
//!
//! ```python
//! def kernel(A: Float[Array, '100 100']):
//!     A[0, 0] = np.sqrt(A[0, 0])
//!     for i in range(1, A.shape[0]):
//!         for j in range(i):
//!             A[i, j] -= np.dot(A[i, :j], A[j, :j])
//!             A[i, j] /= A[j, j]
//!         A[i, i] -= np.dot(A[i, :i], A[i, :i])
//!         A[i, i] = np.sqrt(A[i, i])
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated A
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut a: Array2<Q32> = args.array2();
        a[(0, 0)] = a[(0, 0)].sqrt();
        for i in 1..a.nrows() {
            for j in 0..i {
                let d = np::inner(a.slice(s![i, ..j]), a.slice(s![j, ..j]));
                a[(i, j)] = (a[(i, j)] - d) / a[(j, j)];
            }
            let d = np::inner(a.slice(s![i, ..i]), a.slice(s![i, ..i]));
            a[(i, i)] = (a[(i, i)] - d).sqrt();
        }
        a
    }
}
