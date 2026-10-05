//! DS-1000 case 0304: the longest suffix of whole rows of `ncol = 5`
//! columns, reversed and reshaped into those rows.
//!
//! ```python
//! def kernel(A: Float[Array, "23"]):
//!     ncol = 5
//!     col = ( A.shape[0] // ncol) * ncol
//!     B = A[len(A)-col:][::-1]
//!     B = np.reshape(B, (-1, ncol))
//!     return B
//! ```
use crate::prelude::*;

const LEN: usize = 23;
const NCOL: usize = 5;
/// `col`: the elements that fill whole rows.
const KEPT: usize = LEN / NCOL * NCOL;

pub type Output = Vec<Q32>;

/// `A[len(A)-col:][::-1]`: the last `KEPT` elements, last first; the reshape
/// keeps that order.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(LEN);
    a[LEN - KEPT..].iter().rev().copied().collect()
}
