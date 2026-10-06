//! Pythran grouping: run-length encoding, the first value of every run of
//! equal values and the run's length.
//!
//! ```python
//! def grouping(values: Integer[Array, '5']):
//!     import numpy as np
//!     diff = np.concatenate(([1], np.diff(values)))
//!     idx = np.concatenate((np.where(diff)[0], [len(values)]))
//!     return (values[idx[:-1]], np.diff(idx))
//! ```
use crate::prelude::*;

pub mod grouping {
    use super::*;

    pub type Output = (Vec<i64>, Vec<i64>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let values: Array1<i64> = args.array1();
        let n = values.len();
        // idx: 0, every k where np.diff is nonzero (values[k] != values[k - 1],
        // also under int64 wrapping), then len(values).
        let mut idx = vec![0];
        idx.extend((1..n).filter(|&k| values[k] != values[k - 1]));
        idx.push(n);
        let firsts = idx[..idx.len() - 1].iter().map(|&k| values[k]).collect();
        let lengths = idx.windows(2).map(|w| (w[1] - w[0]) as i64).collect();
        (firsts, lengths)
    }
}
