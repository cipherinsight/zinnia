//! DS-1000 case 0454: every row of an integer matrix divided by its L-infinity
//! norm.
//!
//! ```python
//! def kernel(X: Integer[Array, "5 4"]):
//!     linf = np.abs(X).max(axis = 1)
//!     result = X / linf.reshape(-1, 1)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 4;

pub type Output = Vec<Q32>;

/// The norm is an integer and `/` is NumPy's true division, so each entry is
/// `float(x) / float(linf)`. NumPy gives nan for a row of zeros; that division
/// by zero panics here.
pub fn main(input: &mut impl Input) -> Output {
    let x: [[i64; COLS]; ROWS] = input.read_matrix();
    let mut out = Vec::with_capacity(ROWS * COLS);
    for row in &x {
        let linf = row.iter().map(|v| v.abs()).max().expect("rows are not empty");
        out.extend(row.iter().map(|&v| Q32::from_int(v) / linf));
    }
    out
}
