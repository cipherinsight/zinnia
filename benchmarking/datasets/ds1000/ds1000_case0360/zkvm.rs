//! DS-1000 case 0360: drop the third row, `np.delete(a, 2, axis=0)`.
//!
//! ```python
//! def kernel(a: Float[Array, "4 4"]):
//!     a = np.delete(a, 2, axis = 0)
//!     return a
//! ```
use crate::prelude::*;

const ROWS: usize = 4;
const COLS: usize = 4;
const DELETED: usize = 2;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let mut a: Vec<Q32> = input.read_vec(ROWS * COLS);
    a.drain(DELETED * COLS..(DELETED + 1) * COLS);
    a
}
