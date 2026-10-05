//! DS-1000 case 0337: the diagonal of `a` flipped left to right,
//! `np.diag(np.fliplr(a))`, for a wide matrix.
//!
//! ```python
//! def kernel(a: Float[Array, "8 10"]):
//!     result = np.diag(np.fliplr(a))
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 10;

pub type Output = Vec<Q32>;

/// `np.fliplr` reverses every row, so entry `i` of its diagonal is
/// `a[i][COLS - 1 - i]`; the diagonal has `min(ROWS, COLS) = ROWS` entries.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    (0..ROWS).map(|i| a[i * COLS + (COLS - 1 - i)]).collect()
}
