//! DS-1000 case 0304: the longest suffix of whole rows of `ncol = 5`
//! columns, reversed and reshaped into those rows.
//!
//!     def kernel(A: Float[Array, "23"]):
//!         ncol = 5
//!         col = ( A.shape[0] // ncol) * ncol
//!         B = A[len(A)-col:][::-1]
//!         B = np.reshape(B, (-1, ncol))
//!         return B
// output: [q32]
use crate::q32::Q32;

const LEN: u32 = 23;
const NCOL: u32 = 5;
/// `col`: the elements that fill whole rows.
const KEPT: u32 = LEN / NCOL * NCOL;

/// `A[len(A)-col:][::-1]`: the last `KEPT` elements, last first; the reshape
/// keeps that order.
#[executable]
fn main(A: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == LEN, "A has the wrong length");
    let mut suffix = A.span().slice(LEN - KEPT, KEPT);
    let mut result = array![];
    while let Some(value) = suffix.pop_back() {
        result.append(*value);
    }
    result
}
