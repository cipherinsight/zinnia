//! NPBench polybench/covariance: the `M x M` sample covariance matrix of the
//! columns of `data` (centred in place first).
//!
//! ```python
//! def kernel(M: Integer, float_n: Float, data: Float[Array, '600 500']):
//!     mean = np.mean(data, axis=0)
//!     data -= mean
//!     cov = np.zeros((M, M), dtype=data.dtype)
//!     for i in range(M):
//!         cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
//!     return cov
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let m: i64 = args.scalar();
        let float_n: Q32 = args.scalar();
        let mut data: Array2<Q32> = args.array2();
        let m = usize::try_from(m).expect("negative dimensions are not allowed");

        for mut column in data.columns_mut() {
            let mean = np::mean(&column.to_vec());
            column -= mean;
        }

        let denominator = float_n - Q32::ONE;
        let mut cov = Array2::<Q32>::zeros((m, m));
        for i in 0..m {
            let c = np::vecmat(data.column(i), data.slice(s![.., i..m])).mapv(|v| v / denominator);
            cov.slice_mut(s![i..m, i]).assign(&c);
            cov.slice_mut(s![i, i..m]).assign(&c);
        }
        cov
    }
}
