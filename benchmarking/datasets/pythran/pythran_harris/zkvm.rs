//! Pythran harris: the Harris corner response `det - k * tr^2` of the
//! structure tensor built from forward differences (no smoothing).
//!
//! ```python
//! def harris(I: Float[Array, '4 6']):
//!     m, n = I.shape
//!     dx = (I[1:, :] - I[:m - 1, :])[:, 1:]
//!     dy = (I[:, 1:] - I[:, :n - 1])[1:, :]
//!     A = dx * dx
//!     B = dy * dy
//!     C = dx * dy
//!     tr = A + B
//!     det = A * B - C * C
//!     k = 0.05
//!     return det - k * tr * tr
//! ```
use crate::prelude::*;

/// The kernel's `k = 0.05`, quantized: `round(0.05 * 2^32)`.
const K: Q32 = Q32::from_bits(214_748_365);

pub mod harris {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let img: Array2<Q32> = args.array2();
        let (m, n) = img.dim();
        let shape = (m.saturating_sub(1), n.saturating_sub(1));
        Array2::from_shape_fn(shape, |(i, j)| {
            let dx = img[(i + 1, j + 1)] - img[(i, j + 1)];
            let dy = img[(i + 1, j + 1)] - img[(i + 1, j)];
            let (a, b, c) = (dx * dx, dy * dy, dx * dy);
            let tr = a + b;
            let det = a * b - c * c;
            det - K * tr * tr
        })
    }
}
