//! DS-1000 case 0415: maxima of consecutive bins of `bin_size` values.
//!
//! ```python
//! def kernel(data: Float[Array, "8"]):
//!     bin_size = 4
//!     bin_data_max = data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).max(axis=1)
//!     return bin_data_max
//! ```
use crate::prelude::*;

const N: usize = 8;
const BIN_SIZE: usize = 4;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let data: Vec<Q32> = input.read_vec(N);
    // `chunks_exact` drops a partial last bin, like the slice in the kernel.
    data.chunks_exact(BIN_SIZE)
        .map(|bin| bin.iter().copied().max().expect("bins are not empty"))
        .collect()
}
