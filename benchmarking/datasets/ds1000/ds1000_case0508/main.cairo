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
use core::cmp::{max, min};

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

/// The crop and its data-dependent `(rows, cols)` shape.
#[executable]
fn main(A: Array<i64>) -> (Array<i64>, Array<i64>) {
    assert!(A.len() == ROWS * COLS, "A has the wrong length");
    let a = A.span();
    match bounding_box(a) {
        Some(window) => crop(a, window),
        None => panic!("zero-size array to reduction operation minimum which has no identity"),
    }
}

/// The smallest window that holds every nonzero entry, or `None` if there is none.
fn bounding_box(a: Span<i64>) -> Option<Window> {
    let mut window: Option<Window> = None;
    for i in 0..ROWS {
        for j in 0..COLS {
            if *a.at(i * COLS + j) != 0 {
                window = Some(extend(window, i, j));
            }
        }
    }
    window
}

/// `A[top:bottom+1, left:right+1]` and its shape.
fn crop(a: Span<i64>, w: Window) -> (Array<i64>, Array<i64>) {
    let (height, width) = (w.bottom - w.top + 1, w.right - w.left + 1);
    let mut values = array![];
    for i in w.top..w.bottom + 1 {
        values.append_span(a.slice(i * COLS + w.left, width));
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
