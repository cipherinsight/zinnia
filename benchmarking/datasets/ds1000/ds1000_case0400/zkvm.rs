//! DS-1000 case 0400: an empty array, `np.array([])` (float64, shape `(0,)`).
//!
//! ```python
//! def kernel():
//!     result = np.array([])
//!     return result
//! ```
use crate::prelude::*;

pub type Output = Vec<Q32>;

pub fn main(_input: &mut impl Input) -> Output {
    Vec::new()
}
