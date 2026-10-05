//! DS-1000 case 0310: flat index of the minimum, `a.argmin()`.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         result = a.argmin()
//!         return result
// output: i64
use crate::np::argmin;
use crate::q32::Q32;

#[executable]
fn main(a: Array<Q32>) -> i64 {
    assert!(a.len() == 8 * 6, "a has the wrong length");
    argmin(a.span()).into()
}
