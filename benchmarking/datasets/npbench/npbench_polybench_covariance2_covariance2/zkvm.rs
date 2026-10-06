//! NPBench polybench/covariance2: the sample covariance matrix of the
//! columns of `data` (rows are observations), via `np.cov`.
//!
//! ```python
//! @requires(lambda M, float_n, data: 0 <= M <= 1024)
//! def kernel(M: Integer, float_n: Float, data: Float[Array, '600 500']):
//!     return np.cov(np.transpose(data))
//! ```
//!
//! `np.cov(X)` (variables in the rows of `X`, `ddof = 1`) computes
//! `X -= X.mean(axis=1)[:, None]`, `c = X @ X.T`, `c *= 1 / (n - 1)` with
//! `n` observations: the scale is the rounded reciprocal, multiplied in.
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // unused parameters: float_n
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let m: i64 = args.scalar();
        let data: Array2<Q32> = args.array2();
        // `M` is read only by the precondition.
        assert!((0..=1024).contains(&m), "requires 0 <= M <= 1024");
        let mut x = data.reversed_axes();
        let observations = x.ncols() as i64;
        for mut row in x.rows_mut() {
            let mean = row.sum() / observations;
            row.mapv_inplace(|v| v - mean);
        }
        let scale = Q32::ONE / (observations - 1);
        np::matmul(x.view(), x.t()).mapv(|c| c * scale)
    }
}
