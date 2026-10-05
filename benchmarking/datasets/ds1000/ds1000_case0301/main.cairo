//! DS-1000 case 0301: a vector reshaped into rows of `ncol = 5` columns,
//! `np.reshape(A, (-1, ncol))`.
//!
//!     def kernel(A: Float[Array, "20"]):
//!         ncol = 5
//!         B = np.reshape(A, (-1, ncol))
//!         return B
// output: [q32]
use crate::q32::Q32;

const LEN: u32 = 20;
/// `ncol`; `LEN` is a multiple of it, as NumPy requires.
const NCOL: u32 = 5;

/// A reshape keeps the row-major order, so the `(LEN / NCOL, NCOL)` result
/// holds the elements of `A` as they are.
#[executable]
fn main(A: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == LEN, "A has the wrong length");
    A
}
