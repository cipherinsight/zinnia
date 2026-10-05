//! DS-1000 case 0430: the interval `mean -/+ 3 * std`.
//!
//! ```python
//! def kernel(a: Float[Array, "30"]):
//!     result = (a.mean()-3*a.std(), a.mean()+3*a.std())
//!
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 30;

pub type Output = (Q32, Q32);

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let mean = np::mean(&a);
    let spread = np::std(&a, mean) * 3;
    (mean - spread, mean + spread)
}
