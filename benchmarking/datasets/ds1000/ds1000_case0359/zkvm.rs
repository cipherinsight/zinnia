//! DS-1000 case 0359: drop the third column, `np.delete(a, 2, axis=1)`.
//!
//! ```python
//! def kernel(a: Float[Array, "3 3"]):
//!     a = np.delete(a, 2, axis = 1)
//!     return a
//! ```
use crate::prelude::*;

const ROWS: usize = 3;
const COLS: usize = 3;
const DELETED: usize = 2;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    a.chunks_exact(COLS)
        .flat_map(|row| row[..DELETED].iter().chain(&row[DELETED + 1..]))
        .copied()
        .collect()
}
