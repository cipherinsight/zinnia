//! DS-1000 case 0391: the columns `low` to `high - 1` of a matrix,
//! `a[:, low:high]`.
//!
//! ```python
//! def kernel(a: Float[Array, "20 10"]):
//!     low = 4
//!     high = 7
//!     result = a[:, low:high]
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 20;
const COLS: usize = 10;
const LOW: usize = 4;
const HIGH: usize = 7;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    a.chunks_exact(COLS).flat_map(|row| row[LOW..HIGH].iter().copied()).collect()
}
