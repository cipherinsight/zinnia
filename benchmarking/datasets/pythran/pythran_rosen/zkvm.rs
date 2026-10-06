//! Pythran rosen: the Rosenbrock function of `x`.
//!
//! ```python
//! def rosen(x: Float[Array, '1000']):
//!     t0 = 100 * (x[1:] - x[:-1] ** 2) ** 2
//!     t1 = (1 - x[:-1]) ** 2
//!     return np.sum(t0 + t1)
//! ```
use crate::prelude::*;

pub mod rosen {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        x.windows(2)
            .into_iter()
            .map(|pair| {
                let (xi, xnext) = (pair[0], pair[1]);
                let a = xnext - xi * xi;
                let b = Q32::ONE - xi;
                a * a * 100 + b * b
            })
            .sum()
    }
}
