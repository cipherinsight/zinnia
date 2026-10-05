//! DS-1000 case 0300: `np.percentile(a, p)` with linear interpolation.
//!
//! ```python
//! def kernel(a: Float[Array, "20"], p: Integer):
//!     result = np.percentile(a, p)
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 20;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let a: [Q32; N] = input.read_array();
    let p: i64 = input.read();
    percentile(a, p)
}

/// NumPy's default ("linear") method: the value at rank `p/100 * (n-1)` of
/// the sorted data, interpolated between its two neighbours.
fn percentile(mut a: [Q32; N], p: i64) -> Q32 {
    assert!((0..=100).contains(&p), "percentile outside [0, 100]");
    a.sort_unstable();
    let rank_times_100 = p * (N as i64 - 1);
    let lo = (rank_times_100 / 100) as usize;
    let hi = (lo + 1).min(N - 1);
    let weight = Q32::from_int(rank_times_100 % 100) / 100;
    a[lo] + (a[hi] - a[lo]) * weight
}
