//! NPBench polybench/correlation: the `M x M` correlation matrix of the
//! columns of `data` (standardized in place first).
//!
//! ```python
//! def kernel(M: Integer, float_n: Float, data: Float[Array, '600 500']):
//!     mean = np.mean(data, axis=0)
//!     stddev = np.std(data, axis=0)
//!     stddev[stddev <= 0.1] = 1.0
//!     data -= mean
//!     data /= np.sqrt(float_n) * stddev
//!     corr = np.eye(M, dtype=data.dtype)
//!     for i in range(M - 1):
//!         corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]
//!     return corr
//! ```
use crate::prelude::*;

/// `0.1`, quantized.
const STDDEV_FLOOR: Q32 = Q32::from_bits(429_496_730);

pub mod kernel {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let m: i64 = args.scalar();
        let float_n: Q32 = args.scalar();
        let mut data: Array2<Q32> = args.array2();
        let m = usize::try_from(m).expect("negative dimensions are not allowed");

        let sqrt_n = float_n.sqrt();
        for mut column in data.columns_mut() {
            let values = column.to_vec();
            let mean = np::mean(&values);
            let mut stddev = np::std(&values, mean);
            if stddev <= STDDEV_FLOOR {
                stddev = Q32::ONE;
            }
            let scale = sqrt_n * stddev;
            column.mapv_inplace(|v| (v - mean) / scale);
        }

        let mut corr = Array2::<Q32>::eye(m);
        for i in 0..m.saturating_sub(1) {
            let c = np::vecmat(data.column(i), data.slice(s![.., i + 1..m]));
            corr.slice_mut(s![i + 1..m, i]).assign(&c);
            corr.slice_mut(s![i, i + 1..m]).assign(&c);
        }
        corr
    }
}
