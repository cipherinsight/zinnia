//! DS-1000 case 0444: the entries of `A` strictly inside `(B[0], B[1])` or
//! `(B[1], B[2])`, in order.
//!
//! ```python
//! def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
//!     C = A[np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])]
//!     return C
//! ```
use crate::prelude::*;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [i64; 20] = input.read_array();
    let b: [i64; 3] = input.read_array();
    a.into_iter()
        .filter(|&v| strictly_between(v, b[0], b[1]) || strictly_between(v, b[1], b[2]))
        .collect()
}

fn strictly_between(v: i64, low: i64, high: i64) -> bool {
    low < v && v < high
}
