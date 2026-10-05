//! DS-1000 case 0301: a vector reshaped into rows of `ncol = 5` columns,
//! `np.reshape(A, (-1, ncol))`.
//!
//! ```python
//! def kernel(A: Float[Array, "20"]):
//!     ncol = 5
//!     B = np.reshape(A, (-1, ncol))
//!     return B
//! ```
use crate::prelude::*;

const LEN: usize = 20;
const NCOL: usize = 5;

// NumPy raises unless the rows come out whole.
const _: () = assert!(LEN % NCOL == 0);

pub type Output = Vec<Q32>;

/// A reshape keeps the row-major order, so the `(LEN / NCOL, NCOL)` result
/// holds the elements of `A` as they are.
pub fn main(input: &mut impl Input) -> Output {
    input.read_vec(LEN)
}
