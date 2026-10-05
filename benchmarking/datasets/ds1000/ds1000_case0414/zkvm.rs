//! DS-1000 case 0414: means of consecutive bins of `bin_size` values.
//!
//! ```python
//! def kernel(data: Float[Array, "8"]):
//!     bin_size = 4
//!     bin_data_mean = data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)
//!     return bin_data_mean
//! ```
use crate::prelude::*;

const N: usize = 8;
const BIN_SIZE: usize = 4;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let data: Vec<Q32> = input.read_vec(N);
    // `chunks_exact` drops a partial last bin, like the slice in the kernel.
    data.chunks_exact(BIN_SIZE).map(mean).collect()
}

fn mean(bin: &[Q32]) -> Q32 {
    bin.iter().sum::<Q32>() / bin.len() as i64
}
