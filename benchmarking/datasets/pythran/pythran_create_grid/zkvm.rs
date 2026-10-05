//! Pythran create_grid: all pairs `(x[i], x[j], 0)` of a 1-D array, as an
//! `N*N x 3` array.
//!
//! ```python
//! def create_grid(x: Float[Array, '1000']):
//!     N = x.shape[0]
//!     z = np.zeros((N, N, 3))
//!     z[:, :, 0] = x.reshape(-1, 1)
//!     z[:, :, 1] = x
//!     fast_grid = z.reshape(N * N, 3)
//!     return fast_grid
//! ```
use crate::prelude::*;

pub mod create_grid {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        let n = x.len();
        Array2::from_shape_fn((n * n, 3), |(row, axis)| match axis {
            0 => x[row / n],
            1 => x[row % n],
            _ => Q32::ZERO,
        })
    }
}
