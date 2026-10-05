//! DS-1000 case 0428: the interval `mean -/+ 3 * std`.
//!
//!     def kernel(a: Float[Array, "30"]):
//!         result = (a.mean()-3*a.std(), a.mean()+3*a.std())
//!         return result
// output: (q32, q32)
use crate::np::{mean, std};
use crate::q32::{Q32, Q32Trait};

const N: u32 = 30;

#[executable]
fn main(a: Array<Q32>) -> (Q32, Q32) {
    assert!(a.len() == N, "a has the wrong length");
    let mean = mean(a.span());
    let spread = std(a.span(), mean).mul_int(3);
    (mean - spread, mean + spread)
}
