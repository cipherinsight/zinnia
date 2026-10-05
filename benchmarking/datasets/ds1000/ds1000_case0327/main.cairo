//! DS-1000 case 0327: a vector padded with zeros to `length = 17`.
//!
//!     def kernel(A: Float[Array, "10"]):
//!         length = 17
//!         result = np.pad(A, (0, length-A.shape[0]), 'constant')
//!         return result
// output: [q32]
use crate::q32::{Q32, ZERO};

const LEN: u32 = 10;
const LENGTH: u32 = 17;

/// The pad widths `(0, length - len(A))` append `length - len(A)` zeros.
#[executable]
fn main(A: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == LEN, "A has the wrong length");
    let mut result = A;
    for _ in LEN..LENGTH {
        result.append(ZERO);
    }
    result
}
