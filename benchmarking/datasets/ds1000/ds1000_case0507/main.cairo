//! DS-1000 case 0507: crop an image to the rows and columns that hold a
//! nonzero pixel; an all-zero image gives `np.array([])`.
//!
//!     def kernel(im: Integer[Array, "5 6"]):
//!         mask = im == 0
//!         rows = np.flatnonzero((~mask).sum(axis=1))
//!         cols = np.flatnonzero((~mask).sum(axis=0))
//!         if rows.shape[0] == 0:
//!             result = np.array([])
//!         else:
//!             result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//!         return result
// output: ([dim], [i64])
use crate::np::{bounding_box, crop};

const ROWS: u32 = 5;
const COLS: u32 = 6;

/// The crop `(rows, cols)`, or the empty array of shape `(0,)` (`np.array([])`).
#[executable]
fn main(im: Array<i64>) -> (Array<i64>, Array<i64>) {
    assert!(im.len() == ROWS * COLS, "im has the wrong length");
    let im = im.span();
    match bounding_box(im, COLS, |v| v != 0) {
        Some(window) => crop(im, COLS, window),
        None => (array![0], array![]),
    }
}
