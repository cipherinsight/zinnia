//! DS-1000 case 0419: means of bins of `bin_size` values taken from the end
//! of each row, listed left to right.
//!
//! ```python
//! def kernel(data: Float[Array, "8 6"]):
//!     bin_size = 2
//!     new_data = data[:, ::-1]
//!     bin_data_mean = new_data[:,:(data.shape[1] // bin_size) * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)[:,::-1]
//!     return bin_data_mean
//! ```
//!
//! With 6 columns and `bin_size = 2` no partial bin is dropped, so the result
//! equals plain left-to-right binning (case 0416); the program follows the
//! kernel, which aligns the bins to the end of the row.
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
const BIN_SIZE: usize = 2;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let data: Vec<Q32> = input.read_vec(ROWS * COLS);
    // Binning the reversed rows: `rchunks_exact` takes the bins from the end
    // of a row and drops a partial bin at its start; the final `[:, ::-1]`
    // lists them from left to right again. A bin's values come in forward
    // order, which does not change their (exact) sum.
    data.chunks_exact(COLS)
        .flat_map(|row| row.rchunks_exact(BIN_SIZE).rev().map(np::mean))
        .collect()
}
