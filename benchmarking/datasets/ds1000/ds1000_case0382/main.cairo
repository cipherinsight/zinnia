//! DS-1000 case 0382: the indices that sort `a`, `np.argsort(a)`.
//!
//!     def kernel(a: Float[Array, "100"]):
//!         result = np.argsort(a)
//!         return result
// output: [i64]
use crate::np::argsort;
use crate::q32::Q32;

const N: u32 = 100;

#[executable]
fn main(a: Array<Q32>) -> Array<i64> {
    assert!(a.len() == N, "a has the wrong length");
    let mut order = array![];
    for i in argsort(a.span()) {
        order.append(i.into());
    }
    order
}
