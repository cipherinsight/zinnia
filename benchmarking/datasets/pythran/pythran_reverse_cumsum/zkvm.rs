//! Pythran reverse_cumsum: suffix sums, `out[i] = sum(x[i:])`.
//!
//! ```python
//! def reverse_cumsum(x: Float[Array, '10']):
//!     return np.cumsum(x[::-1])[::-1]
//! ```
use crate::prelude::*;

pub mod reverse_cumsum {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut x: Array1<Q32> = args.array1();
        let mut total = Q32::ZERO;
        for v in x.iter_mut().rev() {
            total += *v;
            *v = total;
        }
        x
    }
}
