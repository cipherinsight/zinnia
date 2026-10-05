//! DS-1000 case 0302: a vector reshaped into `nrow = 5` rows,
//! `np.reshape(A, (nrow, -1))`.
//!
//!     def kernel(A: Float[Array, "20"]):
//!         nrow = 5
//!         B = np.reshape(A, (nrow, -1))
//!         return B
// output: [q32]
use crate::q32::Q32;

const LEN: u32 = 20;
/// `nrow`; `LEN` is a multiple of it, as NumPy requires.
const NROW: u32 = 5;

/// A reshape keeps the row-major order, so the `(NROW, LEN / NROW)` result
/// holds the elements of `A` as they are.
#[executable]
fn main(A: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == LEN, "A has the wrong length");
    A
}
