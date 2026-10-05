//! DS-1000 case 0363: insert an integer into a float array,
//! `np.insert(a, pos, element)`.
//!
//!     def kernel(a: Float[Array, "100"], pos: Integer, element: Integer):
//!         a = np.insert(a, pos, element)
//!         return a
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const N: u32 = 100;

#[executable]
fn main(a: Array<Q32>, pos: i64, element: i64) -> Array<Q32> {
    assert!(a.len() == N, "a has the wrong length");
    let at = insert_position(pos, N);
    let a = a.span();
    let mut result = array![];
    result.append_span(a.slice(0, at));
    // `np.insert` casts the value to the array's dtype.
    result.append(Q32Trait::from_int(element));
    result.append_span(a.slice(at, N - at));
    result
}

/// `np.insert`'s position on an axis of length `len`: anything in
/// `[-len, len]`, negative positions counting from the end; NumPy raises
/// `IndexError` outside that range.
fn insert_position(pos: i64, len: u32) -> u32 {
    let n: i64 = len.into();
    assert!(-n <= pos && pos <= n, "insert position out of range");
    let wrapped = if pos < 0 {
        pos + n
    } else {
        pos
    };
    wrapped.try_into().unwrap()
}
