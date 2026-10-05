//! DS-1000 case 0429: the interval `mean -/+ 2 * std`.
//!
//!     def kernel(a: Float[Array, "30"]):
//!         result = (a.mean()-2*a.std(), a.mean()+2*a.std())
//!         return result
// output: (q32, q32)
use crate::q32::{Q32, Q32Trait, ZERO};

const N: u32 = 30;

#[executable]
fn main(a: Array<Q32>) -> (Q32, Q32) {
    assert!(a.len() == N, "a has the wrong length");
    let mut sum = ZERO;
    for v in a.span() {
        sum = sum + *v;
    }
    let mean = sum.div_int(N.into());
    let mut squares = ZERO;
    for v in a.span() {
        let d = *v - mean;
        squares = squares + d * d;
    }
    let spread = squares.div_int(N.into()).sqrt().mul_int(2);
    (mean - spread, mean + spread)
}
