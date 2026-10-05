//! Pythran fbcorr: a filter-bank correlation of a batch of images,
//! accumulated into `output` in place.
//!
//! ```python
//! def fbcorr(imgs: Float[Array, '8 8 8 8'], filters: Float[Array, '8 8 8 8'], output: Float[Array, '8 8 8 8']):
//!     n_imgs, n_rows, n_cols, n_channels = imgs.shape
//!     n_filters, height, width, n_ch2 = filters.shape
//!     for ii in range(n_imgs):
//!         for rr in range(n_rows - height + 1):
//!             for cc in range(n_cols - width + 1):
//!                 for hh in range(height):
//!                     for ww in range(width):
//!                         for jj in range(n_channels):
//!                             for ff in range(n_filters):
//!                                 imgval = imgs[ii, rr + hh, cc + ww, jj]
//!                                 filterval = filters[ff, hh, ww, jj]
//!                                 output[ii, ff, rr, cc] += imgval * filterval
//! ```
//!
//! Every product is rescaled on its own (scalar `*`), as written.
use crate::prelude::*;

pub mod fbcorr {
    use super::*;

    // outputs: mutated output
    pub type Output = Array4<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let imgs: Array4<Q32> = args.array4();
        let filters: Array4<Q32> = args.array4();
        let mut output: Array4<Q32> = args.array4();
        let (n_imgs, n_rows, n_cols, n_channels) = imgs.dim();
        let (n_filters, height, width, _) = filters.dim();
        // `range` of a negative count is empty.
        let (out_rows, out_cols) = ((n_rows + 1).saturating_sub(height), (n_cols + 1).saturating_sub(width));
        for ii in 0..n_imgs {
            for rr in 0..out_rows {
                for cc in 0..out_cols {
                    for hh in 0..height {
                        for ww in 0..width {
                            for jj in 0..n_channels {
                                let imgval = imgs[(ii, rr + hh, cc + ww, jj)];
                                for ff in 0..n_filters {
                                    output[(ii, ff, rr, cc)] += imgval * filters[(ff, hh, ww, jj)];
                                }
                            }
                        }
                    }
                }
            }
        }
        output
    }
}
