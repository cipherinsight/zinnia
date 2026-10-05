//! DS-1000 case 0381: the indices that sort `a` in decreasing order (the
//! argsort reversed; the final `[:len(a)]` keeps all of it).
//!
//!     def kernel(a: Float[Array, "100"]):
//!         result = np.argsort(a)[::-1][:len(a)]
//!         return result
// output: [i64]
use crate::np::argsort;
use crate::q32::Q32;

const N: u32 = 100;

#[executable]
fn main(a: Array<Q32>) -> Array<i64> {
    assert!(a.len() == N, "a has the wrong length");
    let mut order = argsort(a.span()).span();
    let mut descending = array![];
    while let Some(i) = order.pop_back() {
        descending.append((*i).into());
    }
    descending
}
