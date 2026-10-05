//! DS-1000 case 0339: the diagonal and the anti-diagonal of the leading
//! square block of `a`, stacked as two rows.
//!
//! ```python
//! def kernel(a: Float[Array, "8 10"]):
//!     dim = min(a.shape)
//!     b = a[:dim,:dim]
//!     result = np.vstack((np.diag(b), np.diag(np.fliplr(b))))
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 10;
/// `min(a.shape)`.
const DIM: usize = if ROWS < COLS { ROWS } else { COLS };

pub type Output = Vec<Q32>;

/// `b[i][j]` is `a[i][j]`, so the anti-diagonal of `b` is `a[i][DIM - 1 - i]`.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let diagonal = (0..DIM).map(|i| a[i * COLS + i]);
    let anti_diagonal = (0..DIM).map(|i| a[i * COLS + (DIM - 1 - i)]);
    diagonal.chain(anti_diagonal).collect()
}
