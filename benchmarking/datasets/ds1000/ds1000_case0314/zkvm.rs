//! DS-1000 case 0314: (row, column) of the maximum after the first maximum is
//! overwritten with the minimum, i.e. of the second largest value.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"]):
//!     idx = np.unravel_index(a.argmax(), a.shape)
//!     a[idx] = a.min()
//!     result = np.unravel_index(a.argmax(), a.shape)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub type Output = (i64, i64);

pub fn main(input: &mut impl Input) -> Output {
    let mut a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let first = argmax(&a);
    a[first] = a.iter().copied().min().expect("a is not empty");
    let flat = argmax(&a);
    ((flat / COLS) as i64, (flat % COLS) as i64)
}

/// Index of the first occurrence of the maximum, as NumPy defines it.
fn argmax(values: &[Q32]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > values[best] {
            best = i;
        }
    }
    best
}
