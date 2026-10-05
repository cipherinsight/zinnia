//! DS-1000 case 0415: maxima of consecutive bins of `bin_size` values.
//!
//!     def kernel(data: Float[Array, "8"]):
//!         bin_size = 4
//!         bin_data_max = data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).max(axis=1)
//!         return bin_data_max
// output: [q32]
use crate::q32::Q32;

const N: u32 = 8;
const BIN_SIZE: u32 = 4;
/// A partial last bin is dropped.
const BINS: u32 = N / BIN_SIZE;

#[executable]
fn main(data: Array<Q32>) -> Array<Q32> {
    assert!(data.len() == N, "data has the wrong length");
    let mut maxima = array![];
    for b in 0..BINS {
        maxima.append(max(data.span().slice(b * BIN_SIZE, BIN_SIZE)));
    }
    maxima
}

fn max(bin: Span<Q32>) -> Q32 {
    let mut best = *bin.at(0);
    for v in bin {
        if *v > best {
            best = *v;
        }
    }
    best
}
