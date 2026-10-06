//! NPBench go_fast: add the sum of `tanh` over the diagonal to every entry.
//!
//! ```python
//! def go_fast(a: Float[Array, '2000 2000']):
//!     trace = 0.0
//!     for i in range(a.shape[0]):
//!         trace += np.tanh(a[i, i])
//!     return a + trace
//! ```
use crate::prelude::*;

pub mod go_fast {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: Array2<Q32> = args.array2();
        let trace: Q32 = (0..a.nrows()).map(|i| a[(i, i)].tanh()).sum();
        a + trace
    }
}
