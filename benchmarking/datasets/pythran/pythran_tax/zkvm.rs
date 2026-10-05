//! Pythran tax: the average rate of a piecewise income tax over a set of
//! incomes.
//!
//! ```python
//! def tax(d: Float[Array, '10']):
//!     tax_seg1 = d[d > 256303] * 0.45 - 16164.53
//!     tax_seg2 = d[(d > 54057) & (d <= 256303)] * 0.42 - 8475.44
//!     seg3 = d[(d > 13769) & (d <= 54057)] - 13769
//!     seg4 = d[(d > 8820) & (d <= 13769)] - 8820
//!     prog_seg3 = seg3 * 2.2376e-06 + 0.2397
//!     prog_seg4 = seg4 * 1.00727e-05 + 0.14
//!     return (np.sum(tax_seg1) + np.sum(tax_seg2) + np.sum(seg3 * prog_seg3 + 939.57) + np.sum(seg4 * prog_seg4)) / np.sum(d)
//! ```
use crate::prelude::*;

// The float constants, `round(c * 2^32)`.
const RATE1: Q32 = Q32::from_bits(1_932_735_283); // 0.45
const OFFSET1: Q32 = Q32::from_bits(69_426_127_705_211); // 16164.53
const RATE2: Q32 = Q32::from_bits(1_803_886_264); // 0.42
const OFFSET2: Q32 = Q32::from_bits(36_401_737_619_210); // 8475.44
const SLOPE3: Q32 = Q32::from_bits(9_610); // 2.2376e-06
const BASE3: Q32 = Q32::from_bits(1_029_503_661); // 0.2397
const OFFSET3: Q32 = Q32::from_bits(4_035_422_422_303); // 939.57
const SLOPE4: Q32 = Q32::from_bits(43_262); // 1.00727e-05
const BASE4: Q32 = Q32::from_bits(601_295_421); // 0.14

pub mod tax {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let d: Array1<Q32> = args.array1();
        // The incomes in (lo, hi], and above `lo` for the top segment.
        let segment = |lo: i64, hi: Option<i64>| {
            let (lo, hi) = (Q32::from_int(lo), hi.map(Q32::from_int));
            d.iter()
                .copied()
                .filter(move |&v| lo < v && hi.is_none_or(|hi| v <= hi))
        };
        let tax_seg1: Q32 = segment(256303, None).map(|v| v * RATE1 - OFFSET1).sum();
        let tax_seg2: Q32 = segment(54057, Some(256303))
            .map(|v| v * RATE2 - OFFSET2)
            .sum();
        let tax_seg3: Q32 = segment(13769, Some(54057))
            .map(|v| {
                let seg3 = v - Q32::from_int(13769);
                let prog_seg3 = seg3 * SLOPE3 + BASE3;
                seg3 * prog_seg3 + OFFSET3
            })
            .sum();
        let tax_seg4: Q32 = segment(8820, Some(13769))
            .map(|v| {
                let seg4 = v - Q32::from_int(8820);
                let prog_seg4 = seg4 * SLOPE4 + BASE4;
                seg4 * prog_seg4
            })
            .sum();
        (tax_seg1 + tax_seg2 + tax_seg3 + tax_seg4) / d.sum()
    }
}
