//! Pythran allpairs_distances: squared Euclidean distances between the rows
//! of two integer grids, `X = arange(600 * d).reshape(600, d)` and
//! `Y = arange(200 * d).reshape(200, d)`.
//!
//! ```python
//! def dists(X, Y):
//!     return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])
//!
//! def allpairs_distances(d: Integer):
//!     X = np.arange(600 * d).reshape((600, d))
//!     Y = np.arange(200 * d).reshape((200, d))
//!     return dists(X, Y)
//! ```
use crate::prelude::*;

/// The row counts of `X` and `Y`, fixed by the kernel.
const X_ROWS: usize = 600;
const Y_ROWS: usize = 200;

/// `np.arange(rows * d).reshape((rows, d))`. `d = -1` is NumPy's inferred
/// extent: `arange` of a negative count is empty, so the width is 0.
fn grid(rows: usize, d: i64) -> Array2<i64> {
    let width = if d == -1 {
        0
    } else {
        usize::try_from(d).expect("cannot reshape to a negative width")
    };
    Array2::from_shape_fn((rows, width), |(r, c)| (r * width + c) as i64)
}

pub mod allpairs_distances {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let d: i64 = args.scalar();
        let (x, y) = (grid(X_ROWS, d), grid(Y_ROWS, d));
        // Integer arithmetic wraps like int64.
        Array2::from_shape_fn((Y_ROWS, X_ROWS), |(yi, xi)| {
            x.row(xi)
                .iter()
                .zip(y.row(yi))
                .map(|(&a, &b)| (a - b).wrapping_mul(a - b))
                .fold(0, i64::wrapping_add)
        })
    }
}
