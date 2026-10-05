//! DS-1000 case 0492: the number of pairs that are not close,
//! `(~np.isclose(s1, s2, equal_nan=True)).sum()` with the default tolerances.
//! Q32 values are never NaN, so `equal_nan` changes nothing.
//!
//! ```python
//! def kernel(s1: Float[Array, "6 1"], s2: Float[Array, "6 1"]):
//!     result = (~np.isclose(s1,s2, equal_nan=True)).sum()
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 6;

/// `np.isclose`'s default `rtol = 1e-5`, quantized: `round(1e-5 * 2^32)`.
const RTOL: Q32 = Q32::from_bits(42_950);
/// `np.isclose`'s default `atol = 1e-8`, quantized: `round(1e-8 * 2^32)`.
const ATOL: Q32 = Q32::from_bits(43);

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let s1: Vec<Q32> = input.read_vec(N);
    let s2: Vec<Q32> = input.read_vec(N);
    s1.iter().zip(&s2).filter(|&(&a, &b)| !is_close(a, b)).count() as i64
}

/// `np.isclose(a, b)`: `|a - b| <= atol + rtol * |b|` (not symmetric).
fn is_close(a: Q32, b: Q32) -> bool {
    (a - b).abs() <= ATOL + RTOL * b.abs()
}
