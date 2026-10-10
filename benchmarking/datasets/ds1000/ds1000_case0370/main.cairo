//! DS-1000 case 0370: are all rows of `a` close to its first row?
//!
//!     def kernel(a: Float[Array, "3 3"]):
//!         result = np.isclose(a, a[0], atol=0).all()
//!         return result
// output: bool
use crate::np::{DEFAULT_RTOL, isclose};
use crate::q32::{Q32, ZERO};

const N: u32 = 3;

#[executable]
fn main(a: Array<Q32>) -> bool {
    assert!(a.len() == N * N, "a has the wrong length");
    // Row 0 is trivially close to itself.
    for r in 1..N {
        for c in 0..N {
            if !isclose(*a.at(r * N + c), *a.at(c), DEFAULT_RTOL, ZERO) {
                return false;
            }
        }
    }
    true
}
