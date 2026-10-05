//! DS-1000 case 0314: (row, column) of the maximum after the first maximum is
//! overwritten with the minimum, i.e. of the second largest value.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         idx = np.unravel_index(a.argmax(), a.shape)
//!         a[idx] = a.min()
//!         result = np.unravel_index(a.argmax(), a.shape)
//!         return result
// output: (i64, i64)
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>) -> (i64, i64) {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let first = argmax(a.span());
    let lowest = min(a.span());
    // `a[idx] = a.min()`: a copy of `a` with that one entry replaced.
    let mut replaced = array![];
    for i in 0..a.len() {
        replaced.append(if i == first {
            lowest
        } else {
            *a.at(i)
        });
    }
    let flat = argmax(replaced.span());
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

fn min(values: Span<Q32>) -> Q32 {
    let mut lowest = *values.at(0);
    for v in values {
        if *v < lowest {
            lowest = *v;
        }
    }
    lowest
}
