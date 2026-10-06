//! NPBench polybench/bicg: the two matrix-vector products of BiCG,
//! `(r @ A, A @ p)`.
//!
//! ```python
//! def kernel(A: Float[Array, '5000 4000'], p: Float[Array, '4000'], r: Float[Array, '5000']):
//!     return (r @ A, A @ p)
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = (Array1<Q32>, Array1<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: Array2<Q32> = args.array2();
        let p: Array1<Q32> = args.array1();
        let r: Array1<Q32> = args.array1();
        (
            np::vecmat(r.view(), a.view()),
            np::matvec(a.view(), p.view()),
        )
    }
}
