//! DS-1000 case 0453: every row of an integer matrix divided by its L2 norm.
//!
//! ```python
//! def kernel(X: Integer[Array, "5 4"]):
//!     l2 = np.sqrt((X*X).sum(axis=-1))
//!     result = X / l2.reshape(-1, 1)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5;
const COLS: usize = 4;

pub type Output = Vec<Q32>;

/// `X*X` and its row sums are integers; `np.sqrt` turns the sum into a float.
/// NumPy gives nan for a row of zeros; that division by zero panics here.
pub fn main(input: &mut impl Input) -> Output {
    let x: [[i64; COLS]; ROWS] = input.read_matrix();
    let mut out = Vec::with_capacity(ROWS * COLS);
    for row in &x {
        let squares: i64 = row.iter().map(|v| v * v).sum();
        let l2 = Q32::from_int(squares).sqrt();
        out.extend(row.iter().map(|&v| Q32::from_int(v) / l2));
    }
    out
}
