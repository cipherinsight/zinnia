//! NPBench azimint_hist: mean of `data` over `npt` equal-width bins of
//! `radius`, as the ratio of a weighted and an unweighted `np.histogram`.
//!
//! ```python
//! def azimint_hist(data: Float[Array, '400000'], radius: Float[Array, '400000'], npt: Integer):
//!     histu = np.histogram(radius, npt)[0]
//!     histw = np.histogram(radius, npt, weights=data)[0]
//!     return histw / histu
//! ```
//!
//! An empty bin makes NumPy's ratio `0 / 0 = nan`; Q32 has no nan, so the
//! division panics there.
use crate::prelude::*;

pub mod azimint_hist {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array1<Q32> = args.array1();
        let radius: Array1<Q32> = args.array1();
        let npt: i64 = args.scalar();
        assert!(npt >= 1, "`bins` must be positive, when an integer");
        assert_eq!(
            data.len(),
            radius.len(),
            "weights should have the same shape as a"
        );
        // `np.histogram`'s range: the extremes of `radius`, widened by 0.5 on
        // each side when they coincide; (0, 1) for an empty array.
        let (mut first, mut last) = match (radius.iter().min(), radius.iter().max()) {
            (Some(&lo), Some(&hi)) => (lo, hi),
            _ => (Q32::ZERO, Q32::ONE),
        };
        if first == last {
            let half = Q32::ONE / 2;
            (first, last) = (first - half, last + half);
        }
        let edges = np::linspace(first, last, npt + 1).to_vec();
        let bins = npt as usize;
        // A value falls in the last bin whose left edge it reaches (so the
        // last bin also holds the right edge): the number of interior edges
        // at or below it. One pass fills both histograms.
        let mut counts = vec![0i64; bins];
        let mut sums = vec![Q32::ZERO; bins];
        for (&r, &d) in radius.iter().zip(&data) {
            let bin = edges[1..bins].partition_point(|&edge| edge <= r);
            counts[bin] += 1;
            sums[bin] += d;
        }
        sums.iter().zip(&counts).map(|(&s, &n)| s / n).collect()
    }
}
