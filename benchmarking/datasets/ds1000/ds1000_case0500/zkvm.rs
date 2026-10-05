//! DS-1000 case 0500: `a.reshape(-1, 3)`.
//!
//! ```python
//! def kernel(a: Float[Array, "54"]):
//!     a = a.reshape(-1, 3)
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 54;

pub type Output = Vec<Q32>;

/// A reshape keeps the row-major element order, so the `18 x 3` result is `a`
/// itself.
pub fn main(input: &mut impl Input) -> Output {
    input.read_vec(N)
}
