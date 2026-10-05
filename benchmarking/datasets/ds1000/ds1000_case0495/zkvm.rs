//! DS-1000 case 0495: a matrix padded with zeros at the bottom and right to
//! `shape = (93, 13)`.
//!
//! ```python
//! def kernel(a: Float[Array, "41 13"]):
//!     shape = (93, 13)
//!     result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant')
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 41;
const COLS: usize = 13;
const OUT_ROWS: usize = 93;
const OUT_COLS: usize = 13;

// `a` already has the target width, so only zero rows are added.
const _: () = assert!(OUT_COLS == COLS);

pub type Output = Vec<Q32>;

/// The rows of `a` followed by `OUT_ROWS - ROWS` rows of zeros.
pub fn main(input: &mut impl Input) -> Output {
    let mut result: Vec<Q32> = input.read_vec(ROWS * COLS);
    result.resize(OUT_ROWS * OUT_COLS, Q32::ZERO);
    result
}
