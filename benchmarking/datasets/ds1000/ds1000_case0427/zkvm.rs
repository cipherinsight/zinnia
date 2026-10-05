//! DS-1000 case 0427: the low `m = 11` bits of the XOR of all entries of `a`,
//! most significant first.
//!
//! ```python
//! def kernel(a: Integer[Array, "20"]):
//!     m = 11
//!     res = np.array([0])
//!     for i in a:
//!         res = res ^ i
//!     result = (((res[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
//!     return result
//! ```
//!
//! NumPy's `^` and `&` act on the int64 two's-complement representation, as
//! Rust's do.
use crate::prelude::*;

const N: usize = 20;
const M: u32 = 11;

/// Shape `(1, m)`, flattened row-major.
pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [i64; N] = input.read_array();
    let res = a.iter().fold(0, |acc, &v| acc ^ v);
    (0..M).rev().map(|k| ((res & (1 << k)) > 0) as i64).collect()
}
