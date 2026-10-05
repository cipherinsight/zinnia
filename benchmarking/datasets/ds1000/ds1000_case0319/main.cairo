//! DS-1000 case 0319: reorder the first axis of `a` by the inverse of
//! `permutation`.
//!
//!     def kernel(a: Float[Array, "8 6 8"], permutation: Integer[Array, "8"]):
//!         c = np.empty_like(permutation)
//!         c[permutation] = np.arange(len(permutation))
//!         result = a[c, :, :]
//!         return result
// output: [q32]
use crate::np::inverse_permutation;
use crate::q32::Q32;

const N: u32 = 8;
/// Elements of one `a[k, :, :]`.
const BLOCK: u32 = 6 * 8;

#[executable]
fn main(a: Array<Q32>, permutation: Array<i64>) -> Array<Q32> {
    assert!(a.len() == N * BLOCK && permutation.len() == N, "inputs have the wrong length");
    let a = a.span();
    let mut result = array![];
    for k in inverse_permutation(permutation.span()) {
        result.append_span(a.slice(k * BLOCK, BLOCK));
    }
    result
}
