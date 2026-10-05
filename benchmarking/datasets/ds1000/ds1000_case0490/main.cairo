//! DS-1000 case 0490: per-row thresholding of `arr` with the bounds `n1`, `n2`.
//!
//!     def kernel(arr: Float[Array, "5 50"], n1: Integer[Array, "5"], n2: Integer[Array, "5"]):
//!         for a, t1, t2 in zip(arr, n1, n2):
//!             temp = a.copy()
//!             a[np.where(temp < t1)] = 0
//!             a[np.where(temp >= t2)] = 30
//!             a[np.logical_and(temp >= t1, temp < t2)] += 5
//!         return arr
// output: [q32]
use crate::q32::{Q32, Q32Trait, ZERO};

const ROWS: u32 = 5;
const COLS: u32 = 50;

#[executable]
fn main(arr: Array<Q32>, n1: Array<i64>, n2: Array<i64>) -> Array<Q32> {
    assert!(
        arr.len() == ROWS * COLS && n1.len() == ROWS && n2.len() == ROWS,
        "inputs have the wrong length",
    );
    let mut result = array![];
    for r in 0..ROWS {
        let t1 = Q32Trait::from_int(*n1.at(r));
        let t2 = Q32Trait::from_int(*n2.at(r));
        for v in arr.span().slice(r * COLS, COLS) {
            result.append(threshold(*v, t1, t2));
        }
    }
    result
}

/// The three masked updates of one entry. Every condition tests the original
/// value (`temp`), and a later update overwrites an earlier one: `temp >= t2`
/// wins over `temp < t1` where both hold (`t2 <= v < t1`), and `+= 5` touches
/// exactly the remaining entries.
fn threshold(v: Q32, t1: Q32, t2: Q32) -> Q32 {
    if v >= t2 {
        Q32Trait::from_int(30)
    } else if v < t1 {
        ZERO
    } else {
        v + Q32Trait::from_int(5)
    }
}
