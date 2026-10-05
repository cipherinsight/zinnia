//! DS-1000 case 0490: per-row thresholding of `arr` with the bounds `n1`, `n2`.
//!
//! ```python
//! def kernel(arr: Float[Array, "5 50"], n1: Integer[Array, "5"], n2: Integer[Array, "5"]):
//!     for a, t1, t2 in zip(arr, n1, n2):
//!         temp = a.copy()
//!         a[np.where(temp < t1)] = 0
//!         a[np.where(temp >= t2)] = 30
//!         a[np.logical_and(temp >= t1, temp < t2)] += 5
//!     return arr
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 50;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let mut arr: Vec<Q32> = input.read_vec(ROWS * COLS);
    let n1: [i64; ROWS] = input.read_array();
    let n2: [i64; ROWS] = input.read_array();
    for (row, (&t1, &t2)) in arr.chunks_exact_mut(COLS).zip(n1.iter().zip(&n2)) {
        let (t1, t2) = (Q32::from_int(t1), Q32::from_int(t2));
        for v in row {
            *v = threshold(*v, t1, t2);
        }
    }
    arr
}

/// The three masked updates of one entry. Every condition tests the original
/// value (`temp`), and a later update overwrites an earlier one: `temp >= t2`
/// wins over `temp < t1` where both hold (`t2 <= v < t1`), and `+= 5` touches
/// exactly the remaining entries.
fn threshold(v: Q32, t1: Q32, t2: Q32) -> Q32 {
    if v >= t2 {
        Q32::from_int(30)
    } else if v < t1 {
        Q32::ZERO
    } else {
        v + Q32::from_int(5)
    }
}
