//! NPBench azimint_naive: mean of `data` over each of `npt` equal-width
//! radius bins in `[0, max(radius))`.
//!
//! ```python
//! def azimint_naive(data: Float[Array, '400000'], radius: Float[Array, '400000'], npt: Integer):
//!     rmax = radius.max()
//!     res = np.zeros(npt, dtype=np.float64)
//!     for i in range(npt):
//!         r1 = rmax * i / npt
//!         r2 = rmax * (i + 1) / npt
//!         mask_r12 = np.logical_and(r1 <= radius, radius < r2)
//!         values_r12 = data[mask_r12]
//!         res[i] = values_r12.mean()
//!     return res
//! ```
use crate::prelude::*;

pub mod azimint_naive {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array1<Q32> = args.array1();
        let radius: Array1<Q32> = args.array1();
        let npt: i64 = args.scalar();
        assert!(npt != 0, "requires npt != 0");
        let rmax = *radius.iter().max().expect("radius.max() of an empty array");
        (0..npt)
            .map(|i| {
                let (r1, r2) = (rmax * i / npt, rmax * (i + 1) / npt);
                let in_bin: Vec<Q32> = data
                    .iter()
                    .zip(&radius)
                    .filter(|&(_, &r)| r1 <= r && r < r2)
                    .map(|(&d, _)| d)
                    .collect();
                np::mean(&in_bin)
            })
            .collect()
    }
}
