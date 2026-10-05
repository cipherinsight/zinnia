//! DS-1000 case 0416: means of consecutive bins of `bin_size` values in each
//! row.
//!
//!     def kernel(data: Float[Array, "8 6"]):
//!         bin_size = 2
//!         bin_data_mean = data[:,:(data.shape[1] // bin_size) * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)
//!         return bin_data_mean
// output: [q32]
use crate::np::mean;
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;
const BIN_SIZE: u32 = 2;
/// A partial last bin of a row is dropped.
const BINS: u32 = COLS / BIN_SIZE;

#[executable]
fn main(data: Array<Q32>) -> Array<Q32> {
    assert!(data.len() == ROWS * COLS, "data has the wrong length");
    let mut means = array![];
    for r in 0..ROWS {
        for b in 0..BINS {
            means.append(mean(data.span().slice(r * COLS + b * BIN_SIZE, BIN_SIZE)));
        }
    }
    means
}
