//! DS-1000 case 0431: mask of the entries outside the open interval
//! `mean -/+ 2 * std`.
//!
//!     def kernel(a: Float[Array, "8"]):
//!         interval = (a.mean()-2*a.std(), a.mean()+2*a.std())
//!         result = ~np.logical_and(a>interval[0], a<interval[1])
//!         return result
// output: [bool]
use crate::np::{mean, std};
use crate::q32::{Q32, Q32Trait};

const N: u32 = 8;

#[executable]
fn main(a: Array<Q32>) -> Array<bool> {
    assert!(a.len() == N, "a has the wrong length");
    let mean = mean(a.span());
    let spread = std(a.span(), mean).mul_int(2);
    let low = mean - spread;
    let high = mean + spread;
    let mut outside = array![];
    for v in a {
        outside.append(!(v > low && v < high));
    }
    outside
}
