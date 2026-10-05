//! DS-1000 case 0369: is every row of `a` close to its first entry?
//!
//!     def kernel(a: Float[Array, "3 3"]):
//!         result =np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()
//!         return result
// output: bool
use crate::np::{DEFAULT_RTOL, isclose};
use crate::q32::{Q32, ZERO};

const N: u32 = 3;

#[executable]
fn main(a: Array<Q32>) -> bool {
    assert!(a.len() == N * N, "a has the wrong length");
    // Column 0 is trivially close to itself.
    for r in 0..N {
        let first = *a.at(r * N);
        for c in 1..N {
            if !isclose(*a.at(r * N + c), first, DEFAULT_RTOL, ZERO) {
                return false;
            }
        }
    }
    true
}
