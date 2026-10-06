//! NPBench polybench/mvt: matrix-vector products with `A` and its
//! transpose, accumulated into `x1` and `x2` in place.
//!
//! ```python
//! def kernel(x1: Float[Array, '5500'], x2: Float[Array, '5500'], y_1: Float[Array, '5500'], y_2: Float[Array, '5500'], A: Float[Array, '5500 5500']):
//!     x1 += A @ y_1
//!     x2 += y_2 @ A
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated x1, x2
    pub type Output = (Array1<Q32>, Array1<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut x1: Array1<Q32> = args.array1();
        let mut x2: Array1<Q32> = args.array1();
        let y_1: Array1<Q32> = args.array1();
        let y_2: Array1<Q32> = args.array1();
        let a: Array2<Q32> = args.array2();
        x1 += &np::matvec(a.view(), y_1.view());
        x2 += &np::vecmat(y_2.view(), a.view());
        (x1, x2)
    }
}
