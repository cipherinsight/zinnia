//! DS-1000 case 0417: means of bins of `bin_size` values taken from the end
//! of `data`.
//!
//!     def kernel(data: Float[Array, "8"]):
//!         bin_size = 4
//!         new_data = data[::-1]
//!         bin_data_mean = new_data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)
//!         return bin_data_mean
// output: [q32]
use crate::np::mean;
use crate::q32::Q32;

const N: u32 = 8;
const BIN_SIZE: u32 = 4;
/// A partial bin at the start of `data` (the end of `new_data`) is dropped.
const BINS: u32 = N / BIN_SIZE;

#[executable]
fn main(data: Array<Q32>) -> Array<Q32> {
    assert!(data.len() == N, "data has the wrong length");
    // Bin `b` of the reversed data is `data[N - (b + 1) * BIN_SIZE .. N - b * BIN_SIZE]`
    // backwards; the order does not change the (exact) sum.
    let mut means = array![];
    for b in 0..BINS {
        means.append(mean(data.span().slice(N - (b + 1) * BIN_SIZE, BIN_SIZE)));
    }
    means
}
