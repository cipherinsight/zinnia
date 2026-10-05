//! DS-1000 case 0309: flat index of the maximum, `a.argmax()`.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         result = a.argmax()
//!         return result
// output: i64
use crate::q32::Q32;

/// Index of the first occurrence of the maximum, as NumPy defines it.
#[executable]
fn main(a: Array<Q32>) -> i64 {
    assert!(a.len() == 8 * 6, "a has the wrong length");
    let mut best: u32 = 0;
    for i in 1..a.len() {
        if *a.at(i) > *a.at(best) {
            best = i;
        }
    }
    best.into()
}
