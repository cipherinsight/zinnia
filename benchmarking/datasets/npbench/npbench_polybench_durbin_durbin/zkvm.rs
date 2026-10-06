//! NPBench polybench/durbin: Durbin's algorithm for the Yule-Walker
//! equations (a Toeplitz system given by its first row `r`).
//!
//! ```python
//! def kernel(r: Float[Array, '1000']):
//!     y = np.empty_like(r)
//!     alpha = -r[0]
//!     beta = 1.0
//!     y[0] = -r[0]
//!     for k in range(1, r.shape[0]):
//!         beta *= 1.0 - alpha * alpha
//!         alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
//!         y[:k] += alpha * np.flip(y[:k])
//!         y[k] = alpha
//!     return y
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let r: Array1<Q32> = args.array1();
        let n = r.len();
        let mut y = Array1::zeros(n);
        let mut alpha = -r[0];
        let mut beta = Q32::ONE;
        y[0] = -r[0];
        for k in 1..n {
            beta *= Q32::ONE - alpha * alpha;
            // np.dot(np.flip(r[:k]), y[:k]): one fused dot product.
            let reflected = dot((0..k).map(|i| (r[k - 1 - i], y[i])));
            alpha = -(r[k] + reflected) / beta;
            // The right-hand side `alpha * np.flip(y[:k])` is evaluated before
            // the update.
            let flipped: Vec<Q32> = (0..k).map(|i| alpha * y[k - 1 - i]).collect();
            for (yi, f) in y.iter_mut().zip(flipped) {
                *yi += f;
            }
            y[k] = alpha;
        }
        y
    }
}
