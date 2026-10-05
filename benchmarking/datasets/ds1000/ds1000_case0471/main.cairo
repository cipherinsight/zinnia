//! DS-1000 case 0471: the last entry of the last axis, kept as an axis,
//! `Z[..., -1:]`.
//!
//!     def kernel(Z: Float[Array, "6 2 9 3 7 4"]):
//!         result = Z[..., -1:]
//!         return result
// output: [q32]
use crate::q32::Q32;

/// Elements of `Z` along all axes but the last.
const OUTER: u32 = 6 * 2 * 9 * 3 * 7;
/// Length of the last axis.
const LAST: u32 = 4;

/// The last axis is contiguous, so the result is the last element of every
/// run of `LAST`.
#[executable]
fn main(Z: Array<Q32>) -> Array<Q32> {
    assert!(Z.len() == OUTER * LAST, "Z has the wrong length");
    let mut result = array![];
    for k in 0..OUTER {
        result.append(*Z.at(k * LAST + LAST - 1));
    }
    result
}
