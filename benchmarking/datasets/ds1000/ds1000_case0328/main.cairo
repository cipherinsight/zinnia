//! DS-1000 case 0328: a vector cut to `length = 3`, kept at its size by
//! zeroing the rest.
//!
//!     def kernel(A: Float[Array, "5"]):
//!         length = 3
//!         if length > A.shape[0]:
//!             result = np.pad(A, (0, length-A.shape[0]), 'constant')
//!         else:
//!             result = A.copy()
//!             result[length:] = 0
//!         return result
// output: [q32]
use crate::q32::{Q32, ZERO};

const LEN: u32 = 5;
const LENGTH: u32 = 3;

/// `length <= len(A)`, so the kernel takes the `else` branch: a copy of `A`
/// with every element from index `length` on set to zero.
#[executable]
fn main(A: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == LEN, "A has the wrong length");
    let mut result = array![];
    result.append_span(A.span().slice(0, LENGTH));
    for _ in LENGTH..LEN {
        result.append(ZERO);
    }
    result
}
