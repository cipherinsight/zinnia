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
    let c = inverse_permutation(&permutation);
    c.iter().flat_map(|&k| &a[k * BLOCK..(k + 1) * BLOCK]).copied().collect()
}

/// `c[permutation] = np.arange(n)`. An entry that no index writes would keep
/// `np.empty_like`'s uninitialized value, so `permutation` must hit every
/// position, i.e. be a permutation (negative entries count from the end).
fn inverse_permutation<const LEN: usize>(permutation: &[i64; LEN]) -> [usize; LEN] {
    let mut c = [None; LEN];
    for (i, &p) in permutation.iter().enumerate() {
        c[np_index(p, LEN)] = Some(i);
    }
    c.map(|entry| entry.expect("permutation leaves an entry of c uninitialized"))
}
