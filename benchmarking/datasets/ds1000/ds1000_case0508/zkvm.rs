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
use core::ops::RangeInclusive;

use crate::prelude::*;

const N: usize = 10;

type Matrix = [[i64; N]; N];

/// The crop, with its data-dependent `(rows, cols)` shape.
pub type Output = Shaped<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Matrix = input.read_matrix();
    let (rows, cols) = bounding_box(&a, |v| v != 0)
        .expect("zero-size array to reduction operation minimum which has no identity");
    crop(&a, rows, cols)
}

/// The first to last row and the first to last column that hold an entry
/// where `hit` is true, or `None` if no entry is.
fn bounding_box(
    a: &Matrix,
    hit: impl Fn(i64) -> bool,
) -> Option<(RangeInclusive<usize>, RangeInclusive<usize>)> {
    let mut window: Option<(usize, usize, usize, usize)> = None; // top, bottom, left, right
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if hit(v) {
                window = Some(match window {
                    None => (i, i, j, j),
                    Some((top, _, left, right)) => (top, i, left.min(j), right.max(j)),
                });
            }
        }
    }
    window.map(|(top, bottom, left, right)| (top..=bottom, left..=right))
}

fn crop(a: &Matrix, rows: RangeInclusive<usize>, cols: RangeInclusive<usize>) -> Shaped<i64> {
    let shape = vec![extent(&rows), extent(&cols)];
    let values = a[rows].iter().flat_map(|row| row[cols.clone()].iter().copied()).collect();
    Shaped { shape, values }
}

fn extent(range: &RangeInclusive<usize>) -> i64 {
    (range.end() - range.start() + 1) as i64
}
