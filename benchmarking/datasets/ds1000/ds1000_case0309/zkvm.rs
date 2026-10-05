//! DS-1000 case 0309: flat index of the maximum, `a.argmax()`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"]):
//!     result = a.argmax()
//!     return result
//! ```
use crate::prelude::*;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(8 * 6);
    np::argmax(&a) as i64
}
