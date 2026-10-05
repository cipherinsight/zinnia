//! DS-1000 case 0481: every index `i` with `(x[i], y[i]) == (a, b)`.
//!
//! ```python
//! def kernel(x: Integer[Array, "9"], y: Integer[Array, "9"], a: Integer, b: Integer):
//!     idx_list = ((x == a) & (y == b))
//!     result = idx_list.nonzero()[0]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 9;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let x: [i64; N] = input.read_array();
    let y: [i64; N] = input.read_array();
    let a: i64 = input.read();
    let b: i64 = input.read();
    (0..N).filter(|&i| x[i] == a && y[i] == b).map(|i| i as i64).collect()
}
