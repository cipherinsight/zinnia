//! DS-1000 case 0417: means of bins of `bin_size` values taken from the end
//! of `data`.
//!
//! ```python
//! def kernel(data: Float[Array, "8"]):
//!     bin_size = 4
//!     new_data = data[::-1]
//!     bin_data_mean = new_data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)
//!     return bin_data_mean
//! ```
use crate::prelude::*;

const N: usize = 8;
const BIN_SIZE: usize = 4;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let data: Vec<Q32> = input.read_vec(N);
    // Binning the reversed data: `rchunks_exact` takes the bins from the end
    // and drops a partial bin at the start. A bin's values come in forward
    // order, which does not change their (exact) sum.
    data.rchunks_exact(BIN_SIZE).map(np::mean).collect()
}
