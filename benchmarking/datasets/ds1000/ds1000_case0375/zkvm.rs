//! DS-1000 case 0375: the lowest grade and the grade where the empirical CDF
//! first exceeds `threshold`.
//!
//! ```python
//! @zk_chip
//! def ecdf_result(x):
//!     xs = np.sort(x)
//!     ys = np.arange(1, len(xs)+1)/float(len(xs))
//!     return xs, ys
//!
//! @zk_circuit
//! def kernel(grades: Float[Array, "50"], threshold: Float):
//!     resultx, resulty = ecdf_result(grades)
//!     t = (resulty > threshold).argmax()
//!     low = resultx[0]
//!     high = resultx[t]
//!     return [low, high]
//! ```
use crate::prelude::*;

const N: usize = 50;

/// The list `[low, high]`.
pub type Output = (Q32, Q32);

pub fn main(input: &mut impl Input) -> Output {
    let mut xs: [Q32; N] = input.read_array();
    let threshold: Q32 = input.read();
    xs.sort_unstable();
    // `(resulty > threshold).argmax()`: the first index where the CDF exceeds
    // `threshold`, or 0 if there is none.
    let t = (0..N).find(|&k| ecdf_y(k) > threshold).unwrap_or(0);
    (xs[0], xs[t])
}

/// `resulty[k] = (k + 1) / n`, computed only where it is used.
fn ecdf_y(k: usize) -> Q32 {
    Q32::from_int(k as i64 + 1) / N as i64
}
