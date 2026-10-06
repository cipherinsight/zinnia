//! Pythran compute_subpix_2d_gaussian2: sub-pixel peak displacement from a
//! Gaussian fit of the 3 x 3 neighbourhood of a correlation peak (PIV).
//!
//! ```python
//! def compute_subpix_2d_gaussian2(correl: Float[Array, '4 4'], ix: Integer, iy: Integer):
//!     correl_crop = correl[iy - 1:iy + 2, ix - 1:ix + 2]
//!     tmp = np.where(correl_crop < 0)
//!     for i0, i1 in zip(tmp[0], tmp[1]):
//!         correl_crop[i0, i1] = 1e-06
//!     c10 = 0
//!     c01 = 0
//!     c11 = 0
//!     c20 = 0
//!     c02 = 0
//!     for i in range(1):
//!         for j in range(1):
//!             c10 += (i - 1) * np.log(correl_crop[j, i])
//!             c01 += (j - 1) * np.log(correl_crop[j, i])
//!             c11 += (i - 1) * (j - 1) * np.log(correl_crop[j, i])
//!             c20 += (3 * (i - 1) ** 2 - 2) * np.log(correl_crop[j, i])
//!             c02 += (3 * (j - 1) ** 2 - 2) * np.log(correl_crop[j, i])
//!             c00 = (5 - 3 * (i - 1) ** 2 - 3 * (j - 1) ** 2) * np.log(correl_crop[j, i])
//!     c00, c10, c01, c11, c20, c02 = (c00 / 9, c10 / 6, c01 / 6, c11 / 4, c20 / 6, c02 / 6)
//!     deplx = np.float32((c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11 ** 2))
//!     deply = np.float32((c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11 ** 2))
//!     return (deplx, deply, correl_crop)
//! ```

use crate::prelude::*;

/// `round(1e-6 * 2^32)`, the value that replaces negative correlations.
const FLOOR_VALUE: Q32 = Q32::from_bits(4_295);

pub mod compute_subpix_2d_gaussian2 {
    use super::*;

    pub type Output = (Q32, Q32, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let correl: Array2<Q32> = args.array2();
        let ix: i64 = args.scalar();
        let iy: i64 = args.scalar();
        let (rows, cols) = correl.dim();
        let mut correl_crop = correl
            .slice(s![
                np::slice_range(iy - 1, iy + 2, rows),
                np::slice_range(ix - 1, ix + 2, cols)
            ])
            .to_owned();
        correl_crop.mapv_inplace(|v| if v < Q32::ZERO { FLOOR_VALUE } else { v });

        // Upstream's loops run once (i = j = 0), so only correl_crop[0, 0]
        // enters the fit. c00 is never used after the loops and is left out.
        let (mut c10, mut c01, mut c11, mut c20, mut c02) =
            (Q32::ZERO, Q32::ZERO, Q32::ZERO, Q32::ZERO, Q32::ZERO);
        for i in 0..1i64 {
            for j in 0..1i64 {
                let log = correl_crop[(j as usize, i as usize)].ln();
                c10 += log * (i - 1);
                c01 += log * (j - 1);
                c11 += log * ((i - 1) * (j - 1));
                c20 += log * (3 * (i - 1).pow(2) - 2);
                c02 += log * (3 * (j - 1).pow(2) - 2);
            }
        }
        let (c10, c01, c11, c20, c02) = (c10 / 6, c01 / 6, c11 / 4, c20 / 6, c02 / 6);
        let denominator = c20 * 4 * c02 - c11.powi(2);
        // `np.float32(...)` is the identity: Q32 is the one float type, as in
        // Zinnia.
        let deplx = (c11 * c01 - c10 * 2 * c02) / denominator;
        let deply = (c11 * c10 - c01 * 2 * c20) / denominator;
        (deplx, deply, correl_crop)
    }
}
