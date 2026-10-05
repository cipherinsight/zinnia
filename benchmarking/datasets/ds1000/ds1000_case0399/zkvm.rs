//! DS-1000 case 0399: the second-order recurrence
//! `B[k] = a*A[k] + b*B[k-1] + c*B[k-2]`.
//!
//! ```python
//! def kernel(A: Float[Array, "30"], a: Integer, b: Integer, c: Integer):
//!     B = np.empty(len(A))
//!     for k in range(0, len(B)):
//!         if k == 0:
//!             B[k] = a*A[k]
//!         elif k == 1:
//!             B[k] = a*A[k] + b*B[k-1]
//!         else:
//!             B[k] = a*A[k] + b*B[k-1] + c*B[k-2]
//!     return B
//! ```
use crate::prelude::*;

const N: usize = 30;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let big_a: [Q32; N] = input.read_array();
    let a: i64 = input.read();
    let b: i64 = input.read();
    let c: i64 = input.read();
    let mut big_b = Vec::with_capacity(N);
    big_b.push(big_a[0] * a);
    big_b.push(big_a[1] * a + big_b[0] * b);
    for k in 2..N {
        big_b.push(big_a[k] * a + big_b[k - 1] * b + big_b[k - 2] * c);
    }
    big_b
}
