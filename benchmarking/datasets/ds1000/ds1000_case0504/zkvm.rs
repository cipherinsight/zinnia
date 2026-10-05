//! DS-1000 case 0504: the sum of the elements that `b` picks along the last
//! axis, `np.sum(np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0])`.
//!
//! ```python
//! def kernel(a: Float[Array, "13 13 5"], b: Integer[Array, "13 13"]):
//!     arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//!     result = np.sum(arr)
//!     return result
//! ```
use crate::prelude::*;

const DIM: usize = 13;
const DEPTH: usize = 5;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(DIM * DIM * DEPTH);
    let b: Vec<i64> = input.read_vec(DIM * DIM);
    a.chunks_exact(DEPTH)
        .zip(b)
        .map(|(lane, i)| lane[np_index(i, DEPTH)])
        .sum()
}
