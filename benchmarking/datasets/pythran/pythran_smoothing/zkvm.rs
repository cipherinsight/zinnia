//! Pythran smoothing: exponential smoothing of a time series.
//!
//! ```python
//! def smoothing(x: Float[Array, '100'], alpha: Float):
//!     s = x.copy()
//!     for i in range(1, len(x)):
//!         s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
//!     return s
//! ```
use crate::prelude::*;

pub mod smoothing {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        let alpha: Q32 = args.scalar();
        let mut s = x.clone();
        for i in 1..x.len() {
            s[i] = alpha * x[i] + (Q32::ONE - alpha) * s[i - 1];
        }
        s
    }
}
