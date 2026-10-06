//! DS-1000 case 0452: every row of an integer matrix divided by its L1 norm.
//!
//! ```python
//! def kernel(X: Integer[Array, "5 4"]):
//!     l1 = np.abs(X).sum(axis = 1)
//!     result = X / l1.reshape(-1, 1)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 4;

pub type Output = Vec<Q32>;

/// The norm is an integer and `/` is NumPy's true division, so each entry is
/// `float(x) / float(l1)`. NumPy gives nan for a row of zeros; that division
/// by zero panics here. So do, as in Zinnia (no proof), `|i64::MIN|` and a row
/// sum past int64 (both overflow int64; NumPy wraps them) and a norm of
/// `2^31` or more (`float(l1)` has no Q32 value). Every `|x| <= l1` then has
/// one, so `Q32::from_int(x)` is exact.
pub fn main(input: &mut impl Input) -> Output {
    let x: [[i64; COLS]; ROWS] = input.read_matrix();
    let mut out = Vec::with_capacity(ROWS * COLS);
    for row in &x {
        let l1 = row
            .iter()
            .try_fold(0i64, |sum, v| v.checked_abs().and_then(|m| sum.checked_add(m)))
            .expect("np.abs(X).sum() overflows int64");
        assert!(l1 < 1 << 31, "float(l1) has no Q32 value");
        out.extend(row.iter().map(|&v| Q32::from_int(v) / l1));
    }
    out
}
