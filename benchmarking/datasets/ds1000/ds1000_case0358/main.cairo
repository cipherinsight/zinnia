//! DS-1000 case 0358: the slabs `b[i, :, :]` ordered by the sums of the
//! slabs `a[i, :, :]`.
//!
//!     def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
//!         index = np.argsort(a.sum(axis = (1, 2)))
//!         result = b[index, :, :]
//!         return result
// output: [i64]
use crate::np::argsort;
use crate::q32::{Q32, ZERO};

const DEPTH: u32 = 3;
/// Elements per slab (3 x 3).
const SLAB: u32 = 3 * 3;

#[executable]
fn main(a: Array<Q32>, b: Array<i64>) -> Array<i64> {
    assert!(a.len() == DEPTH * SLAB && b.len() == DEPTH * SLAB, "inputs have the wrong length");
    let mut sums = array![];
    for s in 0..DEPTH {
        let mut sum = ZERO;
        for v in a.span().slice(s * SLAB, SLAB) {
            sum = sum + *v;
        }
        sums.append(sum);
    }
    let mut result = array![];
    for s in argsort(sums.span()) {
        result.append_span(b.span().slice(s * SLAB, SLAB));
    }
    result
}
