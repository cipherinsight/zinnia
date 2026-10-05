//! DS-1000 case 0442: the entries of `A` that do not occur in `B`, in order.
//!
//! ```python
//! def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
//!     C = A[~np.in1d(A,B)]
//!     return C
//! ```
use crate::prelude::*;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [i64; 20] = input.read_array();
    let b: [i64; 3] = input.read_array();
    a.into_iter().filter(|v| !b.contains(v)).collect()
}
