//! DS-1000 case 0463: case 0462 on a column vector: drop the zeros and the
//! repeats of the previous entry, as a `(k, 1)` column.
//!
//! ```python
//! def kernel(a: Integer[Array, "20 1"]):
//!     selection = np.ones((len(a), 1), dtype = bool)
//!     selection[1:] = a[1:] != a[:-1]
//!     selection &= a != 0
//!     result = a[selection].reshape(-1, 1)
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 20;

/// The kept entries; the shape `(k, 1)` follows from their count.
pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<i64> = input.read_vec(N);
    (0..N)
        .filter(|&i| a[i] != 0 && (i == 0 || a[i] != a[i - 1]))
        .map(|i| a[i])
        .collect()
}
