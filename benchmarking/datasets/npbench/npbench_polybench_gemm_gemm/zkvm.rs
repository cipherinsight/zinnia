//! NPBench polybench/gemm: `C = alpha * A @ B + beta * C`, in place.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, C: Float[Array, '1000 1100'], A: Float[Array, '1000 1200'], B: Float[Array, '1200 1100']):
//!     C[:] = alpha * A @ B + beta * C
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
        // `alpha * A @ B` is `(alpha * A) @ B`: `*` and `@` group left to right.
        let product = np::matmul(a.mapv(|v| alpha * v).view(), b.view());
        c.zip_mut_with(&product, |c, &p| *c = p + beta * *c);
        c
    }
}
