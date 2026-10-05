//! DS-1000 case 0510: crop an image to the rows and columns that hold a
//! nonzero pixel; an all-zero image gives `np.array([])`.
//!
//! ```python
//! def kernel(im: Float[Array, "10 10"]):
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

const N: usize = 10;

/// The crop `(rows, cols)`, or the empty array of shape `(0,)`.
pub type Output = Shaped<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let im: [[Q32; N]; N] = input.read_matrix();
    match np::bounding_box(&im, |v| v != Q32::ZERO) {
        Some((rows, cols)) => np::crop(&im, rows, cols),
        None => Shaped { shape: vec![0], values: Vec::new() },
    }
}
