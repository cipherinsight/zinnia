//! DS-1000 case 0471: the last entry of the last axis, kept as an axis,
//! `Z[..., -1:]`.
//!
//! ```python
//! def kernel(Z: Float[Array, "6 2 9 3 7 4"]):
//!     result = Z[..., -1:]
//!     return result
//! ```
use crate::prelude::*;

/// Elements of `Z` along all axes but the last.
const OUTER: usize = 6 * 2 * 9 * 3 * 7;
/// Length of the last axis.
const LAST: usize = 4;

pub type Output = Vec<Q32>;

/// The last axis is contiguous, so the result is the last element of every
/// run of `LAST`.
pub fn main(input: &mut impl Input) -> Output {
    let z: Vec<Q32> = input.read_vec(OUTER * LAST);
    z.chunks_exact(LAST).map(|lane| lane[LAST - 1]).collect()
}
