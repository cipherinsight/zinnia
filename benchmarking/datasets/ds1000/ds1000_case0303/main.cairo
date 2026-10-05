//! DS-1000 case 0303: the longest prefix of whole rows of `ncol = 5`
//! columns, reshaped into those rows.
//!
//!     def kernel(A: Float[Array, "23"]):
//!         ncol = 5
//!         col = ( A.shape[0] // ncol) * ncol
//!         B = A[:col]
//!         B= np.reshape(B, (-1, ncol))
//!         return B
// output: [q32]
use crate::q32::Q32;

const LEN: u32 = 23;
const NCOL: u32 = 5;
/// `col`: the elements that fill whole rows.
const KEPT: u32 = LEN / NCOL * NCOL;

/// `A[:col]`; the reshape keeps the row-major order.
#[executable]
fn main(A: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == LEN, "A has the wrong length");
    let mut result = array![];
    result.append_span(A.span().slice(0, KEPT));
    result
}
