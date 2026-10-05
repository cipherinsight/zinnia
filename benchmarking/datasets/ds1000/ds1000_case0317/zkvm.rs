//! DS-1000 case 0317: a copy of an array, `np.array(a)`.
//!
//! ```python
//! def kernel(a: Float[Array, "10 20 5"]):
//!     result = np.array(a)
//!     return result
//! ```
use crate::prelude::*;

const LEN: usize = 10 * 20 * 5;

pub type Output = Vec<Q32>;

/// `np.array` of an array is a copy with the same shape and values.
pub fn main(input: &mut impl Input) -> Output {
    input.read_vec(LEN)
}
