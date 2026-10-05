//! DS-1000 case 0383: the indices of the `N = 7` largest entries, largest
//! first, `np.argsort(a)[::-1][:N]`.
//!
//!     def kernel(a: Float[Array, "100"]):
//!         N = 7
//!         result = np.argsort(a)[::-1][:N]
//!         return result
// output: [i64]
use crate::np::argsort;
use crate::q32::Q32;

const LEN: u32 = 100;
const N: u32 = 7;

/// A stable argsort, reversed, truncated. Equal values come out by decreasing
/// index; NumPy's default sort is not stable for arrays this long, so its
/// order among ties is an implementation detail.
#[executable]
fn main(a: Array<Q32>) -> Array<i64> {
    assert!(a.len() == LEN, "a has the wrong length");
    let mut order = argsort(a.span()).span();
    let mut result = array![];
    for _ in 0..N {
        result.append((*order.pop_back().unwrap()).into());
    }
    result
}
