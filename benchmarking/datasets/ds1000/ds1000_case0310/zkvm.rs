//! DS-1000 case 0310: flat index of the minimum, `a.argmin()`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"]):
//!     result = a.argmin()
//!     return result
//! ```
use crate::prelude::*;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(8 * 6);
    np::argmin(&a) as i64
}
