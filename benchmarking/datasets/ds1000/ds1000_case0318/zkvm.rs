//! DS-1000 case 0318: reorder the columns of `a` by the inverse of
//! `permutation`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"], permutation: Integer[Array, "6"]):
//!     c = np.empty_like(permutation)
//!     c[permutation] = np.arange(len(permutation))
//!     a = a[:, c]
//!     return a
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let permutation: [i64; COLS] = input.read_array();
    let c = inverse_permutation(&permutation);
    a.chunks_exact(COLS).flat_map(|row| c.iter().map(|&j| row[j])).collect()
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
