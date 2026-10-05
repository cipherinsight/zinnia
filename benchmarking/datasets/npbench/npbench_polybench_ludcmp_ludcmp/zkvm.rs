//! NPBench polybench/ludcmp: in-place LU decomposition of `A` (no pivoting),
//! then forward and back substitution for `A x = b`.
//!
//! ```python
//! def kernel(A: Float[Array, '60 60'], b: Float[Array, '60']):
//!     x = np.zeros_like(b)
//!     y = np.zeros_like(b)
//!     for i in range(A.shape[0]):
//!         for j in range(i):
//!             A[i, j] -= A[i, :j] @ A[:j, j]
//!             A[i, j] /= A[j, j]
//!         for j in range(i, A.shape[0]):
//!             A[i, j] -= A[i, :i] @ A[:i, j]
//!     for i in range(A.shape[0]):
//!         y[i] = b[i] - A[i, :i] @ y[:i]
//!     for i in range(A.shape[0] - 1, -1, -1):
//!         x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
//!     return (x, y)
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = (Array1<Q32>, Array1<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut a: Array2<Q32> = args.array2();
        let b: Array1<Q32> = args.array1();
        let n = a.nrows();
        let mut x = Array1::<Q32>::zeros(b.len());
        let mut y = Array1::<Q32>::zeros(b.len());
        for i in 0..n {
            for j in 0..i {
                let update = np::inner(a.slice(s![i, ..j]), a.slice(s![..j, j]));
                a[(i, j)] -= update;
                let pivot = a[(j, j)];
                a[(i, j)] /= pivot;
            }
            for j in i..n {
                let update = np::inner(a.slice(s![i, ..i]), a.slice(s![..i, j]));
                a[(i, j)] -= update;
            }
        }
        for i in 0..n {
            y[i] = b[i] - np::inner(a.slice(s![i, ..i]), y.slice(s![..i]));
        }
        for i in (0..n).rev() {
            x[i] = (y[i] - np::inner(a.slice(s![i, i + 1..]), x.slice(s![i + 1..]))) / a[(i, i)];
        }
        (x, y)
    }
}
