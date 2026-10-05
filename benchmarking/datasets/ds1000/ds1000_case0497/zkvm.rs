//! DS-1000 case 0497: a matrix padded with the integer `element` at the
//! bottom and right to `shape = (93, 13)`.
//!
//! ```python
//! def kernel(a: Float[Array, "42 10"], element: Integer):
//!     shape = (93, 13)
//!     result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant', constant_values=element)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 42;
const COLS: usize = 10;
const OUT_ROWS: usize = 93;
const OUT_COLS: usize = 13;

pub type Output = Vec<Q32>;

/// Row `r` of `a` becomes the start of output row `r`; the rest is `element`,
/// converted to the float dtype of `a`.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let element: i64 = input.read();
    let mut result = vec![Q32::from_int(element); OUT_ROWS * OUT_COLS];
    for (r, row) in a.chunks_exact(COLS).enumerate() {
        let start = r * OUT_COLS;
        result[start..start + COLS].copy_from_slice(row);
    }
    result
}
