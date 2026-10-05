//! DS-1000 case 0472: the last entry of the first axis, kept as an axis,
//! `a[-1:, ...]`.
//!
//!     def kernel(a: Float[Array, "6 2 9 3 7 4"]):
//!         result = a[-1:,...]
//!         return result
// output: [q32]
use crate::q32::Q32;

const LEADING: u32 = 6;
/// Elements of one `a[k, ...]`.
const BLOCK: u32 = 2 * 9 * 3 * 7 * 4;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == LEADING * BLOCK, "a has the wrong length");
    let mut result = array![];
    result.append_span(a.span().slice((LEADING - 1) * BLOCK, BLOCK));
    result
}
