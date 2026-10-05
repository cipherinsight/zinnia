//! DS-1000 case 0401: an empty array of three empty rows,
//! `np.array([[], [], []])` (float64, shape `(3, 0)`). The shape is committed
//! with the (absent) values because their count does not imply it.
//!
//!     def kernel():
//!         result = np.array([[], [], []])
//!         return result
// output: ([dim], [q32])
use crate::q32::Q32;

#[executable]
fn main() -> (Array<i64>, Array<Q32>) {
    (array![3, 0], array![])
}
