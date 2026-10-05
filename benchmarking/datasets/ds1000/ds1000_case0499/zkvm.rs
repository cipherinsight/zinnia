//! DS-1000 case 0499: a matrix centred in zeros of `shape = (93, 13)`; an odd
//! padding puts the extra row or column at the bottom or right.
//!
//! ```python
//! @zk_chip
//! def to_shape(a, shape):
//!     y_, x_ = shape
//!     y, x = a.shape
//!     y_pad = (y_-y)
//!     x_pad = (x_-x)
//!     return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
//!                         (x_pad//2, x_pad//2 + x_pad%2)),
//!                     mode = 'constant')
//!
//! @zk_circuit
//! def kernel(a: Float[Array, "42 10"]):
//!     shape = (93, 13)
//!     result = to_shape(a, shape)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 42;
const COLS: usize = 10;
const OUT_ROWS: usize = 93;
const OUT_COLS: usize = 13;
/// `y_pad // 2`: zero rows above `a`.
const TOP: usize = (OUT_ROWS - ROWS) / 2;
/// `x_pad // 2`: zero columns left of `a`.
const LEFT: usize = (OUT_COLS - COLS) / 2;

pub type Output = Vec<Q32>;

/// Row `r` of `a` lands in output row `TOP + r` from column `LEFT` on; the
/// rest is zero.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let mut result = vec![Q32::ZERO; OUT_ROWS * OUT_COLS];
    for (r, row) in a.chunks_exact(COLS).enumerate() {
        let start = (TOP + r) * OUT_COLS + LEFT;
        result[start..start + COLS].copy_from_slice(row);
    }
    result
}
