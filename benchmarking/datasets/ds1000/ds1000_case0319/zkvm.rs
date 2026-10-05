//! DS-1000 case 0319: reorder the first axis of `a` by the inverse of
//! `permutation`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6 8"], permutation: Integer[Array, "8"]):
//!     c = np.empty_like(permutation)
//!     c[permutation] = np.arange(len(permutation))
//!     result = a[c, :, :]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 8;
/// Elements of one `a[k, :, :]`.
const BLOCK: usize = 6 * 8;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N * BLOCK);
    let permutation: [i64; N] = input.read_array();
    let c = np::inverse_permutation(&permutation);
    c.iter().flat_map(|&k| &a[k * BLOCK..(k + 1) * BLOCK]).copied().collect()
}
