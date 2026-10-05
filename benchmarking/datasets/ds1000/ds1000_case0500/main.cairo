//! DS-1000 case 0500: `a.reshape(-1, 3)`.
//!
//!     def kernel(a: Float[Array, "54"]):
//!         a = a.reshape(-1, 3)
//!         return a
// output: [q32]
use crate::q32::Q32;

const N: u32 = 54;

/// A reshape keeps the row-major element order, so the `18 x 3` result is `a`
/// itself.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N, "a has the wrong length");
    a
}
