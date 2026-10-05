//! DS-1000 case 0303: the longest prefix of whole rows of `ncol = 5`
//! columns, reshaped into those rows.
//!
//! ```python
//! def kernel(A: Float[Array, "23"]):
//!     ncol = 5
//!     col = ( A.shape[0] // ncol) * ncol
//!     B = A[:col]
//!     B= np.reshape(B, (-1, ncol))
//!     return B
//! ```
use crate::prelude::*;

const LEN: usize = 23;
const NCOL: usize = 5;
/// `col`: the elements that fill whole rows.
const KEPT: usize = LEN / NCOL * NCOL;

pub type Output = Vec<Q32>;

/// `A[:col]`; the reshape keeps the row-major order.
pub fn main(input: &mut impl Input) -> Output {
    let mut a: Vec<Q32> = input.read_vec(LEN);
    a.truncate(KEPT);
    a
}
