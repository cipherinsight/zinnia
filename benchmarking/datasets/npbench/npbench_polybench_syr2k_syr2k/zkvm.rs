//! NPBench polybench/syr2k: symmetric rank-2k update of the lower triangle
//! of `C`, `C := alpha * A @ B.T + alpha * B @ A.T + beta * C`, in place.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, C: Float[Array, '50 50'], A: Float[Array, '50 35'], B: Float[Array, '50 35']):
//!     for i in range(A.shape[0]):
//!         C[i, :i + 1] *= beta
//!         for k in range(A.shape[1]):
//!             C[i, :i + 1] += A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k]
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
        let b: Array2<Q32> = args.array2();
        let (n, m) = a.dim();
        for i in 0..n {
            let mut row = c.slice_mut(s![i, ..=i]);
            row.mapv_inplace(|v| v * beta);
            for k in 0..m {
                let (a_ik, b_ik) = (a[(i, k)], b[(i, k)]);
                for (j, c_ij) in row.iter_mut().enumerate() {
                    // Left to right: `(A[j, k] * alpha) * B[i, k]`.
                    *c_ij += a[(j, k)] * alpha * b_ik + b[(j, k)] * alpha * a_ik;
                }
            }
        }
        c
    }
}
