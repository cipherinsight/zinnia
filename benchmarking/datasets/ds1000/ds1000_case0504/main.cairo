//! DS-1000 case 0504: the sum of the elements that `b` picks along the last
//! axis, `np.sum(np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0])`.
//!
//!     def kernel(a: Float[Array, "13 13 5"], b: Integer[Array, "13 13"]):
//!         arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//!         result = np.sum(arr)
//!         return result
// output: q32
use crate::np::np_index;
use crate::q32::{Q32, ZERO};

const DIM: u32 = 13;
const DEPTH: u32 = 5;

#[executable]
fn main(a: Array<Q32>, b: Array<i64>) -> Q32 {
    assert!(a.len() == DIM * DIM * DEPTH && b.len() == DIM * DIM, "inputs have the wrong shape");
    let mut sum = ZERO;
    for lane in 0..DIM * DIM {
        sum = sum + *a.at(lane * DEPTH + np_index(*b.at(lane), DEPTH));
    }
    sum
}
