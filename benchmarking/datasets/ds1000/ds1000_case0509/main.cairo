//! DS-1000 case 0509: crop an image to the rows and columns that hold a
//! zero pixel; an image without zeros gives `np.array([])`.
//!
//!     def kernel(im: Float[Array, "10 10"]):
//!         mask = im == 0
//!         rows = np.flatnonzero((mask).sum(axis=1))
//!         cols = np.flatnonzero((mask).sum(axis=0))
//!
//!         if rows.shape[0] == 0:
//!             result = np.array([])
//!         else:
//!             result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//!         return result
// output: ([dim], [q32])
use crate::np::{bounding_box, crop};
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 10;
const COLS: u32 = 10;

/// The crop `(rows, cols)`, or the empty array of shape `(0,)` (`np.array([])`).
#[executable]
fn main(im: Array<Q32>) -> (Array<i64>, Array<Q32>) {
    assert!(im.len() == ROWS * COLS, "im has the wrong length");
    let im = im.span();
    match bounding_box(im, COLS, |v| v == ZERO) {
        Some(window) => crop(im, COLS, window),
        None => (array![0], array![]),
    }
}
