//! DS-1000 case 0480: the first index `i` with `(x[i], y[i]) == (a, b)`, or
//! -1 if there is none.
//!
//! ```python
//! def kernel(x: Integer[Array, "8"], y: Integer[Array, "8"], a: Integer, b: Integer):
//!     result = ((x == a) & (y == b)).argmax()
//!     if x[result] != a or y[result] != b:
//!         result = -1
//!     return result
//! ```
//!
//! `argmax` of the match mask is the first match, or 0 when nothing matches;
//! in that case index 0 fails the check and the result is -1.
use crate::prelude::*;

const N: usize = 8;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let x: [i64; N] = input.read_array();
    let y: [i64; N] = input.read_array();
    let a: i64 = input.read();
    let b: i64 = input.read();
    match (0..N).position(|i| x[i] == a && y[i] == b) {
        Some(i) => i as i64,
        None => -1,
    }
}
