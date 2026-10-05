//! Pythran pairwise: Euclidean distances between all pairs of rows.
//!
//! ```python
//! def pairwise(X: Float[Array, '200 100']):
//!     M, N = X.shape
//!     D = np.empty((M, M))
//!     for i in range(M):
//!         for j in range(M):
//!             d = 0.0
//!             for k in range(N):
//!                 tmp = X[i, k] - X[j, k]
//!                 d += tmp * tmp
//!             D[i, j] = np.sqrt(d)
//!     return D
//! ```
use crate::prelude::*;

pub mod pairwise {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array2<Q32> = args.array2();
        let m = x.nrows();
        Array2::from_shape_fn((m, m), |(i, j)| {
            let d: Q32 = x
                .row(i)
                .iter()
                .zip(x.row(j))
                .map(|(&a, &b)| {
                    let tmp = a - b;
                    tmp * tmp
                })
                .sum();
            d.sqrt()
        })
    }
}
