//! DS-1000 case 0349: return the fixed column count of `a`.
// output: i64
use crate::q32::Q32;

#[executable]
fn main(a: Array<Q32>) -> i64 {
    assert!(a.len() == 8 * 6, "a has the wrong length");
    6
}
