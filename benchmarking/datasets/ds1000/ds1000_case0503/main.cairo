//! DS-1000 case 0503: pick one element of every lane along the last axis,
//! `np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]`, i.e.
//! `result[i][j] = a[i][j][b[i][j]]`.
//!
//!     def kernel(a: Float[Array, "13 13 5"], b: Integer[Array, "13 13"]):
//!         result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//!         return result
// output: [q32]
use crate::np::np_index;
use crate::q32::Q32;

const DIM: u32 = 13;
const DEPTH: u32 = 5;

#[executable]
fn main(a: Array<Q32>, b: Array<i64>) -> Array<Q32> {
    assert!(a.len() == DIM * DIM * DEPTH && b.len() == DIM * DIM, "inputs have the wrong shape");
    let mut picked = array![];
    for lane in 0..DIM * DIM {
        picked.append(*a.at(lane * DEPTH + np_index(*b.at(lane), DEPTH)));
    }
    picked
}
