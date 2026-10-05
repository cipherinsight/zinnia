//! DS-1000 case 0328: a vector cut to `length = 3`, kept at its size by
//! zeroing the rest.
//!
//! ```python
//! def kernel(A: Float[Array, "5"]):
//!     length = 3
//!     if length > A.shape[0]:
//!         result = np.pad(A, (0, length-A.shape[0]), 'constant')
//!     else:
//!         result = A.copy()
//!         result[length:] = 0
//!     return result
//! ```
use crate::prelude::*;

const LEN: usize = 5;
const LENGTH: usize = 3;

pub type Output = Vec<Q32>;

/// `length <= len(A)`, so the kernel takes the `else` branch: a copy of `A`
/// with every element from index `length` on set to zero.
pub fn main(input: &mut impl Input) -> Output {
    let mut result: Vec<Q32> = input.read_vec(LEN);
    result[LENGTH..].fill(Q32::ZERO);
    result
}
