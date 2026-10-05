//! DS-1000 case 0492: the number of pairs that are not close,
//! `(~np.isclose(s1, s2, equal_nan=True)).sum()` with the default tolerances.
//! Q32 values are never NaN, so `equal_nan` changes nothing.
//!
//!     def kernel(s1: Float[Array, "6 1"], s2: Float[Array, "6 1"]):
//!         result = (~np.isclose(s1,s2, equal_nan=True)).sum()
//!         return result
// output: i64
use crate::np::{DEFAULT_ATOL, DEFAULT_RTOL, isclose};
use crate::q32::Q32;

const N: u32 = 6;

#[executable]
fn main(s1: Array<Q32>, s2: Array<Q32>) -> i64 {
    assert!(s1.len() == N && s2.len() == N, "inputs have the wrong length");
    let mut not_close = 0;
    for i in 0..N {
        if !isclose(*s1.at(i), *s2.at(i), DEFAULT_RTOL, DEFAULT_ATOL) {
            not_close += 1;
        }
    }
    not_close
}
