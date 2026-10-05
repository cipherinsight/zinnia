//! NPBench polybench/lu: LU decomposition without pivoting, in place (L below
//! the diagonal with a unit diagonal implied, U on and above it).
//!
//! ```python
//! def kernel(A: Float[Array, '60 60']):
//!     for i in range(A.shape[0]):
//!         for j in range(i):
//!             A[i, j] -= A[i, :j] @ A[:j, j]
//!             A[i, j] /= A[j, j]
//!         for j in range(i, A.shape[0]):
//!             A[i, j] -= A[i, :i] @ A[:i, j]
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated A
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut a: Array2<Q32> = args.array2();
        let n = a.nrows();
        for i in 0..n {
            for j in 0..i {
                let sum = np::inner(a.slice(s![i, ..j]), a.slice(s![..j, j]));
                let pivot = a[(j, j)];
                a[(i, j)] = (a[(i, j)] - sum) / pivot;
            }
            for j in i..n {
                let sum = np::inner(a.slice(s![i, ..i]), a.slice(s![..i, j]));
                a[(i, j)] -= sum;
            }
        }
        a
    }
}
