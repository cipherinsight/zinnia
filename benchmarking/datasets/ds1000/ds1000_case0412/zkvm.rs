//! DS-1000 case 0412: keep the non-negative entries, `x[x >= 0]`.
//!
//! ```python
//! def kernel(x: Float[Array, "10"]):
//!     result = x[x >=0]
//!     return result
//! ```
use crate::prelude::*;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let x: Vec<Q32> = input.read_vec(10);
    x.into_iter().filter(|&v| v >= Q32::ZERO).collect()
}
