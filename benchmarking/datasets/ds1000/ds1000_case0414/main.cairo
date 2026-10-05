//! DS-1000 case 0414: means of consecutive bins of `bin_size` values.
//!
//!     def kernel(data: Float[Array, "8"]):
//!         bin_size = 4
//!         bin_data_mean = data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)
//!         return bin_data_mean
// output: [q32]
use crate::np::mean;
use crate::q32::Q32;

const N: u32 = 8;
const BIN_SIZE: u32 = 4;
/// A partial last bin is dropped.
const BINS: u32 = N / BIN_SIZE;

#[executable]
fn main(data: Array<Q32>) -> Array<Q32> {
    assert!(data.len() == N, "data has the wrong length");
    let mut means = array![];
    for b in 0..BINS {
        means.append(mean(data.span().slice(b * BIN_SIZE, BIN_SIZE)));
    }
    means
}
