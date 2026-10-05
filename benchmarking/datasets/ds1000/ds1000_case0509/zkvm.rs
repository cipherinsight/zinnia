//! DS-1000 case 0509: crop an image to the rows and columns that hold a
//! zero pixel; an image without zeros gives `np.array([])`.
//!
//! ```python
//! def kernel(im: Float[Array, "10 10"]):
//!     mask = im == 0
//!     rows = np.flatnonzero((mask).sum(axis=1))
//!     cols = np.flatnonzero((mask).sum(axis=0))
//!
//!     if rows.shape[0] == 0:
//!         result = np.array([])
//!     else:
//!         result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
//!     return result
//! ```
use core::ops::RangeInclusive;

use crate::prelude::*;

const N: usize = 10;

type Image = [[Q32; N]; N];

/// The crop `(rows, cols)`, or the empty array of shape `(0,)`.
pub type Output = Shaped<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let im: Image = input.read_matrix();
    match bounding_box(&im, |v| v == Q32::ZERO) {
        Some((rows, cols)) => crop(&im, rows, cols),
        None => Shaped { shape: vec![0], values: Vec::new() },
    }
}

/// The first to last row and the first to last column that hold a pixel
/// where `hit` is true, or `None` if no pixel is.
fn bounding_box(
    im: &Image,
    hit: impl Fn(Q32) -> bool,
) -> Option<(RangeInclusive<usize>, RangeInclusive<usize>)> {
    let mut window: Option<(usize, usize, usize, usize)> = None; // top, bottom, left, right
    for (i, row) in im.iter().enumerate() {
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

fn crop(im: &Image, rows: RangeInclusive<usize>, cols: RangeInclusive<usize>) -> Shaped<Q32> {
    let shape = vec![extent(&rows), extent(&cols)];
    let values = im[rows].iter().flat_map(|row| row[cols.clone()].iter().copied()).collect();
    Shaped { shape, values }
}

fn extent(range: &RangeInclusive<usize>) -> i64 {
    (range.end() - range.start() + 1) as i64
}
