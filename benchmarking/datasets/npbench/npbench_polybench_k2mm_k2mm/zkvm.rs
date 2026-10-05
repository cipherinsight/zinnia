//! NPBench polybench/k2mm: two chained matrix products,
//! `D = alpha * A @ B @ C + beta * D`, in place.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, A: Float[Array, '800 900'], B: Float[Array, '900 850'], C: Float[Array, '850 950'], D: Float[Array, '800 950']):
//!     D[:] = alpha * A @ B @ C + beta * D
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated D
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let alpha: Q32 = args.scalar();
        let beta: Q32 = args.scalar();
        let a: Array2<Q32> = args.array2();
        let b: Array2<Q32> = args.array2();
        let c: Array2<Q32> = args.array2();
        let mut d: Array2<Q32> = args.array2();
        // `*` and `@` group left to right: `((alpha * A) @ B) @ C`.
        let ab = np::matmul(a.mapv(|v| alpha * v).view(), b.view());
        let abc = np::matmul(ab.view(), c.view());
        d.zip_mut_with(&abc, |d, &p| *d = p + beta * *d);
        d
    }
}
