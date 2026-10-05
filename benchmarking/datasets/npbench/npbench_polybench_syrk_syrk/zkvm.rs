//! NPBench polybench/syrk: symmetric rank-k update of the lower triangle of
//! `C`, `C = alpha * A @ A.T + beta * C`, in place.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, C: Float[Array, '70 70'], A: Float[Array, '70 50']):
//!     for i in range(A.shape[0]):
//!         C[i, :i + 1] *= beta
//!         for k in range(A.shape[1]):
//!             C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated C
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let alpha: Q32 = args.scalar();
        let beta: Q32 = args.scalar();
        let mut c: Array2<Q32> = args.array2();
        let a: Array2<Q32> = args.array2();
        for i in 0..a.nrows() {
            let mut row = c.slice_mut(s![i, ..=i]);
            row.mapv_inplace(|v| v * beta);
            for k in 0..a.ncols() {
                // `alpha * A[i, k] * A[:i + 1, k]` is `(alpha * A[i, k]) * A[:i + 1, k]`.
                let scale = alpha * a[(i, k)];
                row.zip_mut_with(&a.slice(s![..=i, k]), |c, &v| *c += scale * v);
            }
        }
        c
    }
}
