//! NPBench polybench/atax: `A.T @ (A @ x)`, written `A @ x @ A`.
//!
//! ```python
//! def kernel(A: Float[Array, '4000 5000'], x: Float[Array, '5000']):
//!     return A @ x @ A
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: Array2<Q32> = args.array2();
        let x: Array1<Q32> = args.array1();
        // `@` groups left to right: `(A @ x) @ A`, each product rounded once per entry.
        let ax = np::matvec(a.view(), x.view());
        np::vecmat(ax.view(), a.view())
    }
}
