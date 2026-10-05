//! DS-1000 case 0321: (row, column) of the maximum.
//!
//!     def kernel(a: Float[Array, "5 6"]):
//!         result = np.unravel_index(a.argmax(), a.shape)
//!         return result
// output: (i64, i64)
use crate::q32::Q32;

const ROWS: u32 = 5;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>) -> (i64, i64) {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let flat = argmax(a.span());
    ((flat / COLS).into(), (flat % COLS).into())
}

/// Index of the first occurrence of the maximum, as NumPy defines it.
fn argmax(values: Span<Q32>) -> u32 {
    let mut best: u32 = 0;
    for i in 1..values.len() {
        if *values.at(i) > *values.at(best) {
            best = i;
        }
    }
    best
}
