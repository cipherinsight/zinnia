//! DS-1000 case 0441: whether `number` occurs in `a` (Python `in` on an
//! array is `(a == number).any()`).
//!
//! ```python
//! def kernel(a: Integer[Array, "4"], number: Integer):
//!     is_contained = number in a
//!     return is_contained
//! ```
use crate::prelude::*;

pub type Output = bool;

pub fn main(input: &mut impl Input) -> Output {
    let a: [i64; 4] = input.read_array();
    let number: i64 = input.read();
    a.contains(&number)
}
