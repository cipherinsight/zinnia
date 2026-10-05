//! DS-1000 case 0442: the entries of `A` that do not occur in `B`, in order.
//!
//!     def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
//!         C = A[~np.in1d(A,B)]
//!         return C
// output: [i64]
use crate::np::contains;

#[executable]
fn main(A: Array<i64>, B: Array<i64>) -> Array<i64> {
    assert!(A.len() == 20 && B.len() == 3, "inputs have the wrong length");
    let b = B.span();
    let mut kept = array![];
    for v in A {
        if !contains(b, v) {
            kept.append(v);
        }
    }
    kept
}
