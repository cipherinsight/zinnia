//! DS-1000 case 0398: the first-order recurrence `B[k] = a*A[k] + b*B[k-1]`.
//!
//!     def kernel(A: Float[Array, "30"], a: Integer, b: Integer):
//!         B = np.empty(len(A))
//!         for k in range(0, len(B)):
//!             if k == 0:
//!                 B[k] = a*A[k]
//!             else:
//!                 B[k] = a*A[k] + b*B[k-1]
//!         return B
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const N: u32 = 30;

#[executable]
fn main(big_a: Array<Q32>, a: i64, b: i64) -> Array<Q32> {
    assert!(big_a.len() == N, "A has the wrong length");
    let mut big_b = array![];
    let mut previous = big_a.at(0).mul_int(a);
    big_b.append(previous);
    for k in 1..N {
        previous = big_a.at(k).mul_int(a) + previous.mul_int(b);
        big_b.append(previous);
    }
    big_b
}
