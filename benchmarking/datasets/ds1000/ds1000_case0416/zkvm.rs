//! DS-1000 case 0416: means of consecutive bins of `bin_size` values in each
//! row.
//!
//! ```python
//! def kernel(data: Float[Array, "8 6"]):
//!     bin_size = 2
//!     bin_data_mean = data[:,:(data.shape[1] // bin_size) * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)
//!     return bin_data_mean
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
const BIN_SIZE: usize = 2;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let data: Vec<Q32> = input.read_vec(ROWS * COLS);
    // `chunks_exact` drops a partial last bin, like the slice in the kernel.
    data.chunks_exact(COLS).flat_map(|row| row.chunks_exact(BIN_SIZE).map(np::mean)).collect()
}
