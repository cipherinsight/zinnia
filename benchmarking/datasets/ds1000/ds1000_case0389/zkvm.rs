//! DS-1000 case 0389: lay `n` small `nrows` x `ncols` matrices out as a grid
//! in one `h` x `w` matrix, row of matrices by row of matrices.
//!
//! ```python
//! def kernel(a: Float[Array, "6 3 4"]):
//!     h = 6
//!     w = 12
//!     n, nrows, ncols = a.shape
//!     result = a.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w)
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 6;
const NROWS: usize = 3;
const NCOLS: usize = 4;
const H: usize = 6;
const W: usize = 12;
/// Rows of the grid, `h // nrows`.
const GRID_ROWS: usize = H / NROWS;
/// Matrices per grid row, the `-1` of the reshape.
const GRID_COLS: usize = N / GRID_ROWS;

// The grid fills the `(h, w)` result exactly, or NumPy's reshape raises.
const _: () = assert!(GRID_ROWS * NROWS == H && GRID_COLS * NCOLS == W);

pub type Output = Vec<Q32>;

/// After the `swapaxes` the axes are (grid row, row within a matrix, grid
/// column, column within a matrix), so output row `g * NROWS + r` is row `r`
/// of the matrices `g * GRID_COLS ..` side by side.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N * NROWS * NCOLS);
    let mut result = Vec::with_capacity(H * W);
    for g in 0..GRID_ROWS {
        for r in 0..NROWS {
            for matrix in g * GRID_COLS..(g + 1) * GRID_COLS {
                let start = (matrix * NROWS + r) * NCOLS;
                result.extend_from_slice(&a[start..start + NCOLS]);
            }
        }
    }
    result
}
