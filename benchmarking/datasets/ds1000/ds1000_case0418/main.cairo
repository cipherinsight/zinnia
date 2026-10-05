//! DS-1000 case 0418: means of bins of `bin_size` values taken from the end
//! of each row.
//!
//!     def kernel(data: Float[Array, "8 6"]):
//!         bin_size = 2
//!         new_data = data[:, ::-1]
//!         bin_data_mean = new_data[:,:(data.shape[1] // bin_size) * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)
//!         return bin_data_mean
// output: [q32]
use crate::q32::{Q32, Q32Trait, ZERO};

const ROWS: u32 = 8;
const COLS: u32 = 6;
const BIN_SIZE: u32 = 2;
/// A partial bin at the start of a row (the end of the reversed row) is
/// dropped.
const BINS: u32 = COLS / BIN_SIZE;

#[executable]
fn main(data: Array<Q32>) -> Array<Q32> {
    assert!(data.len() == ROWS * COLS, "data has the wrong length");
    // Bin `b` of a reversed row is `row[COLS - (b + 1) * BIN_SIZE .. COLS - b * BIN_SIZE]`
    // backwards; the order does not change the (exact) sum.
    let mut means = array![];
    for r in 0..ROWS {
        for b in 0..BINS {
            let start = r * COLS + COLS - (b + 1) * BIN_SIZE;
            means.append(mean(data.span().slice(start, BIN_SIZE)));
        }
    }
    means
}

fn mean(bin: Span<Q32>) -> Q32 {
    let mut sum = ZERO;
    for v in bin {
        sum = sum + *v;
    }
    sum.div_int(bin.len().into())
}
