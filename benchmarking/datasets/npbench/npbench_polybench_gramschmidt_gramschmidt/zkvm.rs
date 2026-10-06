//! NPBench polybench/gramschmidt: QR decomposition by modified Gram-Schmidt;
//! `A` is overwritten along the way, the result is `(Q, R)`.
//!
//! ```python
//! def kernel(A: Float[Array, '70 60']):
//!     Q = np.zeros_like(A)
//!     R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)
//!     for k in range(A.shape[1]):
//!         nrm = np.dot(A[:, k], A[:, k])
//!         R[k, k] = np.sqrt(nrm)
//!         Q[:, k] = A[:, k] / R[k, k]
//!         for j in range(k + 1, A.shape[1]):
//!             R[k, j] = np.dot(Q[:, k], A[:, j])
//!             A[:, j] -= Q[:, k] * R[k, j]
//!     return (Q, R)
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = (Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut a: Array2<Q32> = args.array2();
        let n = a.ncols();
        let mut q = Array2::<Q32>::zeros(a.dim());
        let mut r = Array2::<Q32>::zeros((n, n));
        for k in 0..n {
            let nrm = np::inner(a.column(k), a.column(k));
            r[(k, k)] = nrm.sqrt();
            let r_kk = r[(k, k)];
            Zip::from(q.column_mut(k))
                .and(a.column(k))
                .for_each(|q, &a| *q = a / r_kk);
            for j in k + 1..n {
                r[(k, j)] = np::inner(q.column(k), a.column(j));
                let r_kj = r[(k, j)];
                Zip::from(a.column_mut(j))
                    .and(q.column(k))
                    .for_each(|a, &q| *a -= q * r_kj);
            }
        }
        (q, r)
    }
}
