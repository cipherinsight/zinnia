//! DS-1000 case 0491: the number of pairs that are not close,
//! `(~np.isclose(s1, s2)).sum()` with the default tolerances.
//!
//! ```python
//! def kernel(s1: Float[Array, "6 1"], s2: Float[Array, "6 1"]):
//!     result = (~np.isclose(s1,s2)).sum()
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 6;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let s1: Vec<Q32> = input.read_vec(N);
    let s2: Vec<Q32> = input.read_vec(N);
    let close = |a, b| np::isclose(a, b, np::DEFAULT_RTOL, np::DEFAULT_ATOL);
    s1.iter().zip(&s2).filter(|&(&a, &b)| !close(a, b)).count() as i64
}
