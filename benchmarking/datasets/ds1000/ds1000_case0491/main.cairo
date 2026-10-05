//! DS-1000 case 0491: the number of pairs that are not close,
//! `(~np.isclose(s1, s2)).sum()` with the default tolerances.
//!
//!     def kernel(s1: Float[Array, "6 1"], s2: Float[Array, "6 1"]):
//!         result = (~np.isclose(s1,s2)).sum()
//!         return result
// output: i64
use crate::q32::{Q32, Q32Trait};

const N: u32 = 6;

/// `np.isclose`'s default `rtol = 1e-5`, quantized: `round(1e-5 * 2^32)`.
const RTOL: Q32 = Q32 { bits: 42950 };
/// `np.isclose`'s default `atol = 1e-8`, quantized: `round(1e-8 * 2^32)`.
const ATOL: Q32 = Q32 { bits: 43 };

#[executable]
fn main(s1: Array<Q32>, s2: Array<Q32>) -> i64 {
    assert!(s1.len() == N && s2.len() == N, "inputs have the wrong length");
    let mut not_close = 0;
    for i in 0..N {
        if !is_close(*s1.at(i), *s2.at(i)) {
            not_close += 1;
        }
    }
    not_close
}

/// `np.isclose(a, b)`: `|a - b| <= atol + rtol * |b|` (not symmetric).
fn is_close(a: Q32, b: Q32) -> bool {
    (a - b).abs() <= ATOL + RTOL * b.abs()
}
