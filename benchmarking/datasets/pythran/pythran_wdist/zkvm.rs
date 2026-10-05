//! Pythran wdist: weighted Euclidean distances between the columns of `A`
//! and the columns of `B`, with per-row weights from the columns of `W`.
//!
//! ```python
//! def slow_wdist(A: Float[Array, '2 3'], B: Float[Array, '2 3'], W: Float[Array, '2 3']):
//!     k, m = A.shape
//!     _, n = B.shape
//!     D = np.zeros((m, n))
//!     for ii in range(m):
//!         for jj in range(n):
//!             wdiff = (A[:, ii] - B[:, jj]) / W[:, ii]
//!             D[ii, jj] = np.sqrt((wdiff ** 2).sum())
//!     return D
//! ```
use crate::prelude::*;

pub mod slow_wdist {
    use super::*;

    pub type Output = Array2<Q32>;

    #[allow(non_snake_case)]
    pub fn main(args: &mut Args<impl Input>) -> Output {
        let A: Array2<Q32> = args.array2();
        let B: Array2<Q32> = args.array2();
        let W: Array2<Q32> = args.array2();
        Array2::from_shape_fn((A.ncols(), B.ncols()), |(ii, jj)| {
            // The column operations broadcast like NumPy's.
            let wdiff = (&A.column(ii) - &B.column(jj)) / &W.column(ii);
            wdiff.iter().map(|&v| v * v).sum::<Q32>().sqrt()
        })
    }
}
