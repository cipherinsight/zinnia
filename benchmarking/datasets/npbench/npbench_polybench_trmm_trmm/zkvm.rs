//! NPBench polybench/trmm: triangular matrix product
//! `B = alpha * (B + strictly-lower(A)^T B)`, in place.
//!
//! ```python
//! def kernel(alpha: Float, A: Float[Array, '65 65'], B: Float[Array, '65 80']):
//!     for i in range(B.shape[0]):
//!         for j in range(B.shape[1]):
//!             B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
//!     B *= alpha
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated B
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let alpha: Q32 = args.scalar();
        let a: Array2<Q32> = args.array2();
        let mut b: Array2<Q32> = args.array2();
        let (rows, cols) = b.dim();
        for i in 0..rows {
            for j in 0..cols {
                // Rows below i are not updated yet: B[i + 1:, j] is still the input.
                let below = np::inner(a.slice(s![i + 1.., i]), b.slice(s![i + 1.., j]));
                b[(i, j)] += below;
            }
        }
        b.mapv_inplace(|v| v * alpha);
        b
    }
}
