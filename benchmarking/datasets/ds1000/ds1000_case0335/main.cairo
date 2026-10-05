//! DS-1000 case 0335: element-wise maximum of three arrays.
//!
//!     def kernel(a: Float[Array, "50"], b: Float[Array, "50"], c: Float[Array, "50"]):
//!         result = np.max([a, b, c], axis=0)
//!         return result
// output: [q32]
use core::cmp::max;
use crate::q32::Q32;

const N: u32 = 50;

#[executable]
fn main(a: Array<Q32>, b: Array<Q32>, c: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N && b.len() == N && c.len() == N, "inputs have the wrong length");
    let mut result = array![];
    for i in 0..N {
        result.append(max(max(*a.at(i), *b.at(i)), *c.at(i)));
    }
    result
}
