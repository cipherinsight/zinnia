//! DS-1000 case 0431: mask of the entries outside the open interval
//! `mean -/+ 2 * std`.
//!
//! ```python
//! def kernel(a: Float[Array, "8"]):
//!     interval = (a.mean()-2*a.std(), a.mean()+2*a.std())
//!     result = ~np.logical_and(a>interval[0], a<interval[1])
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 8;

pub type Output = Vec<bool>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let mean = np::mean(&a);
    let spread = np::std(&a, mean) * 2;
    let (low, high) = (mean - spread, mean + spread);
    a.iter().map(|&v| !(v > low && v < high)).collect()
}
