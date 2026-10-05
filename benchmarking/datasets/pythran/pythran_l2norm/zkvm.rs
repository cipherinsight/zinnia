//! Pythran l2norm: the Euclidean norm of every row.
//!
//! ```python
//! def l2_norm(x: Float[Array, '100 100']):
//!     return np.sqrt(np.sum(np.abs(x) ** 2, 1))
//! ```
//!
//! `np.abs(x) ** 2` squares elementwise before the sum (not a contraction),
//! so every square is rescaled on its own.
use crate::prelude::*;

pub mod l2_norm {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array2<Q32> = args.array2();
        x.rows().into_iter().map(|row| row.iter().map(|v| v.abs().powi(2)).sum::<Q32>().sqrt()).collect()
    }
}
