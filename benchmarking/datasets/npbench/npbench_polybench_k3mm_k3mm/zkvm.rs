//! NPBench polybench/k3mm: three chained matrix products.
//!
//! ```python
//! def kernel(A: Float[Array, '800 900'], B: Float[Array, '900 850'], C: Float[Array, '850 1000'], D: Float[Array, '1000 950']):
//!     return A @ B @ C @ D
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: Array2<Q32> = args.array2();
        let b: Array2<Q32> = args.array2();
        let c: Array2<Q32> = args.array2();
        let d: Array2<Q32> = args.array2();
        // `@` groups left to right: ((A @ B) @ C) @ D, each product rescaled.
        let ab = np::matmul(a.view(), b.view());
        let abc = np::matmul(ab.view(), c.view());
        np::matmul(abc.view(), d.view())
    }
}
