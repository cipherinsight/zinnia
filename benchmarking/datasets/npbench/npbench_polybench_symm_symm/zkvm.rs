//! NPBench polybench/symm: `C = alpha * A @ B + beta * C` for a symmetric `A`
//! stored in its lower triangle, in place on `C`.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, C: Float[Array, '40 50'], A: Float[Array, '40 40'], B: Float[Array, '40 50']):
//!     temp2 = np.empty((C.shape[1],), dtype=C.dtype)
//!     C *= beta
//!     for i in range(C.shape[0]):
//!         for j in range(C.shape[1]):
//!             C[:i, j] += alpha * B[i, j] * A[i, :i]
//!             temp2[j] = B[:i, j] @ A[i, :i]
//!         C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
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
        let (rows, cols) = c.dim();
        c.mapv_inplace(|v| v * beta);
        let mut temp2 = Array1::<Q32>::zeros(cols);
        for i in 0..rows {
            let a_row = a.slice(s![i, ..i]);
            for j in 0..cols {
                let scale = alpha * b[(i, j)];
                c.slice_mut(s![..i, j]).zip_mut_with(&a_row, |c, &a| *c += scale * a);
                temp2[j] = np::inner(b.slice(s![..i, j]), a_row);
            }
            let a_ii = a[(i, i)];
            for j in 0..cols {
                c[(i, j)] += alpha * b[(i, j)] * a_ii + alpha * temp2[j];
            }
        }
        c
    }
}
