//! DS-1000 case 0389: lay `n` small `nrows` x `ncols` matrices out as a grid
//! in one `h` x `w` matrix, row of matrices by row of matrices.
//!
//!     def kernel(a: Float[Array, "6 3 4"]):
//!         h = 6
//!         w = 12
//!         n, nrows, ncols = a.shape
//!         result = a.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w)
//!         return result
// output: [q32]
use crate::q32::Q32;

const N: u32 = 6;
const NROWS: u32 = 3;
const NCOLS: u32 = 4;
/// `h`; the grid fills the `(h, w)` result exactly (`w = 12 = GRID_COLS * NCOLS`).
const H: u32 = 6;
/// Rows of the grid, `h // nrows`.
const GRID_ROWS: u32 = H / NROWS;
/// Matrices per grid row, the `-1` of the reshape.
const GRID_COLS: u32 = N / GRID_ROWS;

/// After the `swapaxes` the axes are (grid row, row within a matrix, grid
/// column, column within a matrix), so output row `g * NROWS + r` is row `r`
/// of the matrices `g * GRID_COLS ..` side by side.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N * NROWS * NCOLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    for g in 0..GRID_ROWS {
        for r in 0..NROWS {
            for matrix in g * GRID_COLS..(g + 1) * GRID_COLS {
                result.append_span(a.slice((matrix * NROWS + r) * NCOLS, NCOLS));
            }
        }
    }
    result
}
