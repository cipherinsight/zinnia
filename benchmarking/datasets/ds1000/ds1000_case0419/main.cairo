//! DS-1000 case 0419: means of bins of `bin_size` values taken from the end
//! of each row, listed left to right.
//!
//!     def kernel(data: Float[Array, "8 6"]):
//!         bin_size = 2
//!         new_data = data[:, ::-1]
//!         bin_data_mean = new_data[:,:(data.shape[1] // bin_size) * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)[:,::-1]
//!         return bin_data_mean
//!
//! With 6 columns and `bin_size = 2` no partial bin is dropped, so the result
//! equals plain left-to-right binning (case 0416); the program follows the
//! kernel, which aligns the bins to the end of the row.
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
    // backwards; the final `[:, ::-1]` lists the bins from `b = BINS - 1` down
    // to 0. The order within a bin does not change its (exact) sum.
    let mut means = array![];
    for r in 0..ROWS {
        let mut b = BINS;
        while b > 0 {
            b -= 1;
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
