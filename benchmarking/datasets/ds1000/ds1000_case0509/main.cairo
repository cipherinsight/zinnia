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
use core::cmp::{max, min};
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 10;
const COLS: u32 = 10;

/// A block of cells: rows `top..=bottom`, columns `left..=right`.
#[derive(Copy, Drop)]
struct Window {
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
}

/// The crop `(rows, cols)`, or the empty array of shape `(0,)` (`np.array([])`).
#[executable]
fn main(im: Array<Q32>) -> (Array<i64>, Array<Q32>) {
    assert!(im.len() == ROWS * COLS, "im has the wrong length");
    let im = im.span();
    match bounding_box(im) {
        Some(window) => crop(im, window),
        None => (array![0], array![]),
    }
}

/// The smallest window that holds every zero pixel, or `None` if there is none.
fn bounding_box(im: Span<Q32>) -> Option<Window> {
    let mut window: Option<Window> = None;
    for i in 0..ROWS {
        for j in 0..COLS {
            if *im.at(i * COLS + j) == ZERO {
                window = Some(extend(window, i, j));
            }
        }
    }
    window
}

/// `im[top:bottom+1, left:right+1]` and its shape.
fn crop(im: Span<Q32>, w: Window) -> (Array<i64>, Array<Q32>) {
    let (height, width) = (w.bottom - w.top + 1, w.right - w.left + 1);
    let mut values = array![];
    for i in w.top..w.bottom + 1 {
        values.append_span(im.slice(i * COLS + w.left, width));
    }
    (array![height.into(), width.into()], values)
}

/// `window` grown to hold cell `(i, j)`; rows are scanned in order, so `i` is
/// its new last row.
fn extend(window: Option<Window>, i: u32, j: u32) -> Window {
    match window {
        None => Window { top: i, bottom: i, left: j, right: j },
        Some(w) => Window { top: w.top, bottom: i, left: min(w.left, j), right: max(w.right, j) },
    }
}
