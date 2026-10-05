//! DS-1000 case 0401: an empty array of three empty rows,
//! `np.array([[], [], []])` (float64, shape `(3, 0)`). The shape is committed
//! with the (absent) values because their count does not imply it.
//!
//! ```python
//! def kernel():
//!     result = np.array([[], [], []])
//!     return result
//! ```
use crate::prelude::*;

pub type Output = Shaped<Q32>;

pub fn main(_input: &mut impl Input) -> Output {
    Shaped { shape: vec![3, 0], values: Vec::new() }
}
