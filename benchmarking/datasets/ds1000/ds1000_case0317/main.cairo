//! DS-1000 case 0317: a copy of an array, `np.array(a)`.
//!
//!     def kernel(a: Float[Array, "10 20 5"]):
//!         result = np.array(a)
//!         return result
// output: [q32]
use crate::q32::Q32;

const LEN: u32 = 10 * 20 * 5;

/// `np.array` of an array is a copy with the same shape and values.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == LEN, "a has the wrong length");
    a
}
