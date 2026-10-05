//! DS-1000 case 0425: the low `m = 11` bits of every entry of `a`, most
//! significant first.
//!
//! ```python
//! def kernel(a: Integer[Array, "20"]):
//!     m = 11
//!     result = (((a[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
//!     return result
//! ```
//!
//! NumPy's `&` acts on the int64 two's-complement representation, as Rust's
//! does, so a negative entry gives the low bits of that representation.
use crate::prelude::*;

const N: usize = 20;
const M: u32 = 11;

/// Shape `(20, m)`, flattened row-major.
pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [i64; N] = input.read_array();
    a.iter().flat_map(|&v| (0..M).rev().map(move |k| ((v & (1 << k)) > 0) as i64)).collect()
}
