//! Pythran specialconvolve: the 3x3 box sum of each interior cell of `a`
//! minus nine times the cell (a discrete Laplacian-like filter), on integers.
//!
//! ```python
//! def specialconvolve(a: Integer[Array, '100 100']):
//!     rowconvol = a[1:-1, :] + a[:-2, :] + a[2:, :]
//!     colconvol = rowconvol[:, 1:-1] + rowconvol[:, :-2] + rowconvol[:, 2:] - 9 * a[1:-1, 1:-1]
//!     return colconvol
//! ```
use crate::prelude::*;

pub mod specialconvolve {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: Array2<i64> = args.array2();
        let (rows, cols) = a.dim();
        let rowconvol = Array2::from_shape_fn((rows.saturating_sub(2), cols), |(i, j)| {
            a[(i + 1, j)] + a[(i, j)] + a[(i + 2, j)]
        });
        Array2::from_shape_fn((rows.saturating_sub(2), cols.saturating_sub(2)), |(i, j)| {
            rowconvol[(i, j + 1)] + rowconvol[(i, j)] + rowconvol[(i, j + 2)]
                - 9 * a[(i + 1, j + 1)]
        })
    }
}
