//! DS-1000 case 0508: crop a matrix to the bounding box of its nonzero
//! entries (`np.argwhere`, then the per-axis min and max). An all-zero matrix
//! makes `B.min(0)` reduce an empty array, which NumPy refuses.
//!
//! ```python
//! def kernel(A: Integer[Array, "10 10"]):
//!     B = np.argwhere(A)
//!     (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
//!     result = A[ystart:ystop, xstart:xstop]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 10;

/// The crop, with its data-dependent `(rows, cols)` shape.
pub type Output = Shaped<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[i64; N]; N] = input.read_matrix();
    let (rows, cols) = np::bounding_box(&a, |v| v != 0)
        .expect("zero-size array to reduction operation minimum which has no identity");
    np::crop(&a, rows, cols)
}
