//! DS-1000 case 0334: element-wise mean of three arrays.
//!
//!     def kernel(a: Float[Array, "50"], b: Float[Array, "50"], c: Float[Array, "50"]):
//!         result = np.mean([a, b, c], axis=0)
//!         return result
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const N: u32 = 50;

#[executable]
fn main(a: Array<Q32>, b: Array<Q32>, c: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N && b.len() == N && c.len() == N, "inputs have the wrong length");
    let mut mean = array![];
    for i in 0..N {
        mean.append((*a.at(i) + *b.at(i) + *c.at(i)).div_int(3));
    }
    mean
}
