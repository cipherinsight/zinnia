//! DS-1000 case 0363: insert an integer into a float array,
//! `np.insert(a, pos, element)`.
//!
//! ```python
//! def kernel(a: Float[Array, "100"], pos: Integer, element: Integer):
//!     a = np.insert(a, pos, element)
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 100;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let pos: i64 = input.read();
    let element: i64 = input.read();
    let at = np::insert_position(pos, N);
    let mut result = Vec::with_capacity(N + 1);
    result.extend_from_slice(&a[..at]);
    // `np.insert` casts the value to the array's dtype.
    result.push(Q32::from_int(element));
    result.extend_from_slice(&a[at..]);
    result
}
