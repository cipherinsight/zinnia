//! DS-1000 case 0472: the last entry of the first axis, kept as an axis,
//! `a[-1:, ...]`.
//!
//! ```python
//! def kernel(a: Float[Array, "6 2 9 3 7 4"]):
//!     result = a[-1:,...]
//!     return result
//! ```
use crate::prelude::*;

const LEADING: usize = 6;
/// Elements of one `a[k, ...]`.
const BLOCK: usize = 2 * 9 * 3 * 7 * 4;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(LEADING * BLOCK);
    a[(LEADING - 1) * BLOCK..].to_vec()
}
