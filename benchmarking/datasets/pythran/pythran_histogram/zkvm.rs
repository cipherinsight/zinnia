//! Pythran histogram: counts of the data in `bin_width` equal-width bins over
//! `[min, max]`, the maximum counted in the last bin.
//!
//! ```python
//! def histogram(data: Float[Array, '10000'], bin_width: Integer):
//!     lower_bound, upper_bound = (min(data), max(data))
//!     out_data = [0] * (1 + bin_width)
//!     for i in data:
//!         out_data[int(bin_width * (i - lower_bound) / (upper_bound - lower_bound))] += 1
//!     out_data[-2] += out_data[-1]
//!     out_data.pop()
//!     return out_data
//! ```
use crate::prelude::*;

pub mod histogram {
    use super::*;

    pub type Output = Vec<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array1<Q32> = args.array1();
        let bin_width: i64 = args.scalar();
        let lower_bound = *data.iter().min().expect("min() arg is an empty sequence");
        let upper_bound = *data.iter().max().expect("max() arg is an empty sequence");
        // `[0] * n` is empty for a negative n.
        let mut out_data = vec![0i64; usize::try_from(1 + bin_width).unwrap_or(0)];
        let bins = out_data.len();
        for &i in &data {
            // int * float is exact; the quotient is rescaled; int() truncates.
            let bin = ((i - lower_bound) * bin_width / (upper_bound - lower_bound)).trunc_to_int();
            out_data[np_index(bin, bins)] += 1;
        }
        let last = out_data[np_index(-1, bins)];
        out_data[np_index(-2, bins)] += last;
        out_data.pop();
        out_data
    }
}
