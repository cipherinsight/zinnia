//! DS-1000 case 0393: the columns `low` to `high - 1` of a matrix,
//! `a[:, low:high]`.
//!
//! ```python
//! def kernel(a: Float[Array, "20 10"]):
//!     low = 4
//!     high = 7
//!     high = min(high, a.shape[1])
//!     result = a[:, low:high]
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 20;
const COLS: usize = 10;
const LOW: usize = 4;
/// `min(high, a.shape[1])` with `high = 7`.
const HIGH: usize = if 7 < COLS { 7 } else { COLS };

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    a.chunks_exact(COLS).flat_map(|row| row[LOW..HIGH].iter().copied()).collect()
}
