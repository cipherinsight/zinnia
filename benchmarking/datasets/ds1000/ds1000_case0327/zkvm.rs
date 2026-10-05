//! DS-1000 case 0327: a vector padded with zeros to `length = 17`.
//!
//! ```python
//! def kernel(A: Float[Array, "10"]):
//!     length = 17
//!     result = np.pad(A, (0, length-A.shape[0]), 'constant')
//!     return result
//! ```
use crate::prelude::*;

const LEN: usize = 10;
const LENGTH: usize = 17;

pub type Output = Vec<Q32>;

/// The pad widths `(0, length - len(A))` append `length - len(A)` zeros.
pub fn main(input: &mut impl Input) -> Output {
    let mut result: Vec<Q32> = input.read_vec(LEN);
    result.resize(LENGTH, Q32::ZERO);
    result
}
