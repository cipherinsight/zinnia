//! DS-1000 case 0365: insert an integer into a float array,
//! `np.insert(a, pos, element)`.
//!
//! ```python
//! def kernel(a: Float[Array, "100"], pos: Integer, element: Integer):
//!     a = np.insert(a, pos, element)
//!
//!
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 100;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let pos: i64 = input.read();
    let element: i64 = input.read();
    let at = insert_position(pos, N);
    let mut result = Vec::with_capacity(N + 1);
    result.extend_from_slice(&a[..at]);
    // `np.insert` casts the value to the array's dtype.
    result.push(Q32::from_int(element));
    result.extend_from_slice(&a[at..]);
    result
}

/// `np.insert`'s position on an axis of length `len`: anything in
/// `[-len, len]`, negative positions counting from the end; NumPy raises
/// `IndexError` outside that range.
fn insert_position(pos: i64, len: usize) -> usize {
    let n = len as i64;
    assert!(-n <= pos && pos <= n, "insert position out of range");
    (if pos < 0 { pos + n } else { pos }) as usize
}
