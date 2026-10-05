//! DS-1000 case 0437: mask of the entries equal to their row's minimum.
//!
//! ```python
//! def kernel(a: Float[Array, "10 5"]):
//!     mask = (a.min(axis=1,keepdims=1) == a)
//!     return mask
//! ```
use crate::prelude::*;

const ROWS: usize = 10;
const COLS: usize = 5;

pub type Output = Vec<bool>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let mut mask = Vec::with_capacity(ROWS * COLS);
    for row in a.chunks_exact(COLS) {
        let min = row.iter().copied().min().expect("rows are not empty");
        mask.extend(row.iter().map(|&v| v == min));
    }
    mask
}
