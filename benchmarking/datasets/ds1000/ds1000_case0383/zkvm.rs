//! DS-1000 case 0383: the indices of the `N = 7` largest entries, largest
//! first, `np.argsort(a)[::-1][:N]`.
//!
//! ```python
//! def kernel(a: Float[Array, "100"]):
//!     N = 7
//!     result = np.argsort(a)[::-1][:N]
//!     return result
//! ```
use crate::prelude::*;

const LEN: usize = 100;
const N: usize = 7;

pub type Output = Vec<i64>;

/// A stable argsort, reversed, truncated. Equal values come out by decreasing
/// index; NumPy's default sort is not stable for arrays this long, so its
/// order among ties is an implementation detail.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(LEN);
    np::argsort(&a).iter().rev().take(N).map(|&i| i as i64).collect()
}
