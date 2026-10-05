//! DS-1000 case 0503: pick one element of every lane along the last axis,
//! `np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]`, i.e.
//! `result[i][j] = a[i][j][b[i][j]]`.
//!
//! ```python
//! def kernel(a: Float[Array, "13 13 5"], b: Integer[Array, "13 13"]):
//!     result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//!     return result
//! ```
use crate::prelude::*;

const DIM: usize = 13;
const DEPTH: usize = 5;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(DIM * DIM * DEPTH);
    let b: Vec<i64> = input.read_vec(DIM * DIM);
    a.chunks_exact(DEPTH)
        .zip(b)
        .map(|(lane, i)| lane[np_index(i, DEPTH)])
        .collect()
}
