//! DS-1000 case 0309: flat index of the maximum, `a.argmax()`.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         result = a.argmax()
//!         return result
// output: i64
use crate::np::argmax;
use crate::q32::Q32;

#[executable]
fn main(a: Array<Q32>) -> i64 {
    assert!(a.len() == 8 * 6, "a has the wrong length");
    argmax(a.span()).into()
}
