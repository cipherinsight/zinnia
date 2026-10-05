//! DS-1000 case 0375: the lowest grade and the grade where the empirical CDF
//! first exceeds `threshold`.
//!
//!     @zk_chip
//!     def ecdf_result(x):
//!         xs = np.sort(x)
//!         ys = np.arange(1, len(xs)+1)/float(len(xs))
//!         return xs, ys
//!
//!     @zk_circuit
//!     def kernel(grades: Float[Array, "50"], threshold: Float):
//!         resultx, resulty = ecdf_result(grades)
//!         t = (resulty > threshold).argmax()
//!         low = resultx[0]
//!         high = resultx[t]
//!         return [low, high]
// output: (q32, q32)
use crate::np::sort;
use crate::q32::{Q32, Q32Trait};

const N: u32 = 50;

/// Returns the list `[low, high]`.
#[executable]
fn main(grades: Array<Q32>, threshold: Q32) -> (Q32, Q32) {
    assert!(grades.len() == N, "grades has the wrong length");
    let xs = sort(grades.span()).span();
    // `(resulty > threshold).argmax()`: the first index where the CDF exceeds
    // `threshold`, or 0 if there is none.
    let mut t = 0;
    while t < N && ecdf_y(t) <= threshold {
        t += 1;
    }
    if t == N {
        t = 0;
    }
    (*xs.at(0), *xs.at(t))
}

/// `resulty[k] = (k + 1) / n`, computed only where it is used.
fn ecdf_y(k: u32) -> Q32 {
    Q32Trait::from_int((k + 1).into()).div_int(N.into())
}
