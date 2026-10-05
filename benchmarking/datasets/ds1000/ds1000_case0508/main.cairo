//! DS-1000 case 0508: crop a matrix to the bounding box of its nonzero
//! entries (`np.argwhere`, then the per-axis min and max). An all-zero matrix
//! makes `B.min(0)` reduce an empty array, which NumPy refuses.
//!
//!     def kernel(A: Integer[Array, "10 10"]):
//!         B = np.argwhere(A)
//!         (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
//!         result = A[ystart:ystop, xstart:xstop]
//!         return result
// output: ([dim], [i64])
use crate::np::{bounding_box, crop};

const ROWS: u32 = 10;
const COLS: u32 = 10;

/// The crop and its data-dependent `(rows, cols)` shape.
#[executable]
fn main(A: Array<i64>) -> (Array<i64>, Array<i64>) {
    assert!(A.len() == ROWS * COLS, "A has the wrong length");
    let a = A.span();
    match bounding_box(a, COLS, |v| v != 0) {
        Some(window) => crop(a, COLS, window),
        None => panic!("zero-size array to reduction operation minimum which has no identity"),
    }
}
