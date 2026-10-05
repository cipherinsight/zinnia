//! DS-1000 case 0507: crop an image to the rows and columns that hold a
//! nonzero pixel; an all-zero image gives `np.array([])`.
//!
//! ```python
//! def kernel(im: Integer[Array, "5 6"]):
//!     mask = im == 0
//!     rows = np.flatnonzero((~mask).sum(axis=1))
//!     cols = np.flatnonzero((~mask).sum(axis=0))
//!     if rows.shape[0] == 0:
//!         result = np.array([])
//!     else:
//!         result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 6;

/// The crop `(rows, cols)`, or the empty array of shape `(0,)`.
pub type Output = Shaped<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let im: [[i64; COLS]; ROWS] = input.read_matrix();
    match np::bounding_box(&im, |v| v != 0) {
        Some((rows, cols)) => np::crop(&im, rows, cols),
        None => Shaped { shape: vec![0], values: Vec::new() },
    }
}
