//! DS-1000 case 0412: keep the non-negative entries, `x[x >= 0]`.
//!
//!     def kernel(x: Float[Array, "10"]):
//!         result = x[x >=0]
//!         return result
// output: [q32]
use crate::q32::{Q32, ZERO};

#[executable]
fn main(x: Array<Q32>) -> Array<Q32> {
    assert!(x.len() == 10, "x has the wrong length");
    let mut kept = array![];
    for v in x {
        if v >= ZERO {
            kept.append(v);
        }
    }
    kept
}
