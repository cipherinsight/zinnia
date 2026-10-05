//! DS-1000 case 0302: a vector reshaped into `nrow = 5` rows,
//! `np.reshape(A, (nrow, -1))`.
//!
//! ```python
//! def kernel(A: Float[Array, "20"]):
//!     nrow = 5
//!     B = np.reshape(A, (nrow, -1))
//!     return B
//! ```
use crate::prelude::*;

const LEN: usize = 20;
const NROW: usize = 5;

// NumPy raises unless the rows come out whole.
const _: () = assert!(LEN % NROW == 0);

pub type Output = Vec<Q32>;

/// A reshape keeps the row-major order, so the `(NROW, LEN / NROW)` result
/// holds the elements of `A` as they are.
pub fn main(input: &mut impl Input) -> Output {
    input.read_vec(LEN)
}
