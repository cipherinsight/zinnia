//! DS-1000 case 0341: the elements of `X` in row-major order (`X.flat`).
//!
//! ```python
//! def kernel(X: Integer[Array, "5 6"]):
//!     result = []
//!     for value in X.flat:
//!         result.append(value)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 6;

pub type Output = Vec<i64>;

/// `X.flat` visits the elements in row-major order, the order the input
/// holds them in, so the list is `X` itself.
pub fn main(input: &mut impl Input) -> Output {
    input.read_vec(ROWS * COLS)
}
