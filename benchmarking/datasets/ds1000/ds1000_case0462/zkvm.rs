//! DS-1000 case 0462: drop the zeros and the repeats of the previous entry
//! (keep the first entry of each run of equal values, unless it is zero).
//!
//! ```python
//! def kernel(a: Integer[Array, "20"]):
//!     selection = np.ones(len(a), dtype = bool)
//!     selection[1:] = a[1:] != a[:-1]
//!     selection &= a != 0
//!     result = a[selection]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 20;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [i64; N] = input.read_array();
    (0..N)
        .filter(|&i| a[i] != 0 && (i == 0 || a[i] != a[i - 1]))
        .map(|i| a[i])
        .collect()
}
