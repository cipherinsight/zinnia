//! NPBench polybench/gesummv: `alpha * A @ x + beta * B @ x`.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, A: Float[Array, '2000 2000'], B: Float[Array, '2000 2000'], x: Float[Array, '2000']):
//!     return alpha * A @ x + beta * B @ x
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = Array1<Q32>;

    #[allow(non_snake_case)]
    pub fn main(args: &mut Args<impl Input>) -> Output {
        let alpha: Q32 = args.scalar();
        let beta: Q32 = args.scalar();
        let A: Array2<Q32> = args.array2();
        let B: Array2<Q32> = args.array2();
        let x: Array1<Q32> = args.array1();
        // `alpha * A @ x` is `(alpha * A) @ x`: `*` and `@` group left to right.
        let ax = np::matvec(A.mapv(|v| alpha * v).view(), x.view());
        let bx = np::matvec(B.mapv(|v| beta * v).view(), x.view());
        ax + bx
    }
}
