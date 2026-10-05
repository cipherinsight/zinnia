//! DS-1000 case 0361: drop the first and third columns,
//! `np.delete(a, [0, 2], axis=1)`.
//!
//! ```python
//! def kernel(a: Float[Array, "6 6"]):
//!     temp = np.array([0, 2])
//!     a = np.delete(a, temp, axis = 1)
//!     return a
//! ```
use crate::prelude::*;

const ROWS: usize = 6;
const COLS: usize = 6;
const DELETED: [usize; 2] = [0, 2];

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let kept: Vec<usize> = (0..COLS).filter(|j| !DELETED.contains(j)).collect();
    a.chunks_exact(COLS).flat_map(|row| kept.iter().map(|&j| row[j])).collect()
}
