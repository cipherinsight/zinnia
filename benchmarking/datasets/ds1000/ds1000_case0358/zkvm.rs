//! DS-1000 case 0358: the slabs `b[i, :, :]` ordered by the sums of the
//! slabs `a[i, :, :]`.
//!
//! ```python
//! def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
//!     index = np.argsort(a.sum(axis = (1, 2)))
//!     result = b[index, :, :]
//!     return result
//! ```
use crate::prelude::*;

const DEPTH: usize = 3;
/// Elements per slab (3 x 3).
const SLAB: usize = 3 * 3;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(DEPTH * SLAB);
    let b: Vec<i64> = input.read_vec(DEPTH * SLAB);
    let sums: Vec<Q32> = a.chunks_exact(SLAB).map(|slab| slab.iter().sum()).collect();
    let mut index: Vec<usize> = (0..DEPTH).collect();
    index.sort_by_key(|&i| sums[i]);
    let mut result = Vec::with_capacity(DEPTH * SLAB);
    for i in index {
        result.extend_from_slice(&b[i * SLAB..(i + 1) * SLAB]);
    }
    result
}
