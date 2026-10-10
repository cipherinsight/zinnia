//! DS-1000 case 0291: return the fixed shape of `a`.
// output: (i64, i64, i64, i64)
use crate::q32::Q32;

#[executable]
fn main(a: Array<Q32>) -> (i64, i64, i64, i64) {
    assert!(a.len() == 2 * 5 * 5 * 2, "a has the wrong length");
    (2, 5, 5, 2)
}
