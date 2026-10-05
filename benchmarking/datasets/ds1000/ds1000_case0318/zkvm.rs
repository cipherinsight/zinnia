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
    let c = np::inverse_permutation(&permutation);
    a.chunks_exact(COLS).flat_map(|row| c.iter().map(|&j| row[j])).collect()
}
