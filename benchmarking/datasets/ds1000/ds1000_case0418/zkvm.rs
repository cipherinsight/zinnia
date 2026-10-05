//! DS-1000 case 0418: means of bins of `bin_size` values taken from the end
//! of each row.
//!
//! ```python
//! def kernel(data: Float[Array, "8 6"]):
//!     bin_size = 2
//!     new_data = data[:, ::-1]
//!     bin_data_mean = new_data[:,:(data.shape[1] // bin_size) * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)
//!     return bin_data_mean
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
const BIN_SIZE: usize = 2;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let data: Vec<Q32> = input.read_vec(ROWS * COLS);
    // Binning the reversed rows: `rchunks_exact` takes the bins from the end
    // of a row and drops a partial bin at its start. A bin's values come in
    // forward order, which does not change their (exact) sum.
    data.chunks_exact(COLS).flat_map(|row| row.rchunks_exact(BIN_SIZE).map(mean)).collect()
}

fn mean(bin: &[Q32]) -> Q32 {
    bin.iter().sum::<Q32>() / bin.len() as i64
}
