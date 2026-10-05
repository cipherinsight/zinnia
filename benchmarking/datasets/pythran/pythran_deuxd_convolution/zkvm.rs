//! Pythran deuxd_convolution: 2-D convolution with clamped indices. As
//! upstream, both indices are clamped to `[0, sw[0]]` (the kernel's size, not
//! the image's), so only the top-left corner of `x` is ever read.
//!
//! ```python
//! def clamp(i, offset, maxval):
//!     j = max(0, i + offset)
//!     return min(j, maxval)
//!
//! def conv(x: Float[Array, '300 300'], weights: Float[Array, '5 5']):
//!     sx = x.shape
//!     sw = weights.shape
//!     result = np.zeros_like(x)
//!     for i in range(sx[0]):
//!         for j in range(sx[1]):
//!             for ii in range(sw[0]):
//!                 for jj in range(sw[1]):
//!                     idx = (clamp(i, ii - sw[0] // 2, sw[0]), clamp(j, jj - sw[0] // 2, sw[0]))
//!                     result[i, j] += x[idx] * weights[ii, jj]
//!     return result
//! ```
use crate::prelude::*;

/// `clamp(i, offset, maxval)`: `i + offset` limited to `[0, maxval]`.
fn clamp(i: usize, offset: i64, maxval: usize) -> usize {
    (i as i64 + offset).clamp(0, maxval as i64) as usize
}

pub mod conv {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array2<Q32> = args.array2();
        let weights: Array2<Q32> = args.array2();
        let (rows, cols) = x.dim();
        let (kh, kw) = weights.dim();
        let half = (kh / 2) as i64;
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            let mut acc = Q32::ZERO;
            for ii in 0..kh {
                for jj in 0..kw {
                    let idx = (
                        clamp(i, ii as i64 - half, kh),
                        clamp(j, jj as i64 - half, kh),
                    );
                    acc += x[idx] * weights[(ii, jj)];
                }
            }
            acc
        })
    }
}
