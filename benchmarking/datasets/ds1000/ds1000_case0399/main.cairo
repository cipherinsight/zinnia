//! DS-1000 case 0399: the second-order recurrence
//! `B[k] = a*A[k] + b*B[k-1] + c*B[k-2]`.
//!
//!     def kernel(A: Float[Array, "30"], a: Integer, b: Integer, c: Integer):
//!         B = np.empty(len(A))
//!         for k in range(0, len(B)):
//!             if k == 0:
//!                 B[k] = a*A[k]
//!             elif k == 1:
//!                 B[k] = a*A[k] + b*B[k-1]
//!             else:
//!                 B[k] = a*A[k] + b*B[k-1] + c*B[k-2]
//!         return B
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const N: u32 = 30;

#[executable]
fn main(big_a: Array<Q32>, a: i64, b: i64, c: i64) -> Array<Q32> {
    assert!(big_a.len() == N, "A has the wrong length");
    let mut big_b = array![];
    let mut before_previous = big_a.at(0).mul_int(a);
    let mut previous = big_a.at(1).mul_int(a) + before_previous.mul_int(b);
    big_b.append(before_previous);
    big_b.append(previous);
    for k in 2..N {
        let next = big_a.at(k).mul_int(a) + previous.mul_int(b) + before_previous.mul_int(c);
        big_b.append(next);
        before_previous = previous;
        previous = next;
    }
    big_b
}
