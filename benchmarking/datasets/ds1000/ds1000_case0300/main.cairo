//! DS-1000 case 0300: `np.percentile(a, p)` with linear interpolation.
//!
//!     def kernel(a: Float[Array, "20"], p: Integer):
//!         result = np.percentile(a, p)
//!         return result
// output: q32
use crate::np::sort;
use crate::q32::{Q32, Q32Trait};

const N: u32 = 20;

/// NumPy's default ("linear") method: the value at rank `p/100 * (n-1)` of the
/// sorted data, interpolated between its two neighbours.
#[executable]
fn main(a: Array<Q32>, p: i64) -> Q32 {
    assert!(a.len() == N, "a has the wrong length");
    assert!(0 <= p && p <= 100, "percentile outside [0, 100]");
    let sorted = sort(a.span());
    let rank_times_100 = p * (N - 1).into();
    let lo: u32 = (rank_times_100 / 100).try_into().unwrap();
    let hi = if lo + 1 < N {
        lo + 1
    } else {
        N - 1
    };
    let weight = Q32Trait::from_int(rank_times_100 % 100).div_int(100);
    let low = *sorted.at(lo);
    low + (*sorted.at(hi) - low) * weight
}
