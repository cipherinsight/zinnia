//! DS-1000 case 0322: the (row, column) pairs where `a` attains its minimum,
//! in row-major order.
//!
//! ```python
//! def kernel(a: Integer[Array, "2 2"]):
//!     result = np.argwhere(a == np.min(a))
//!     return result
//! ```
//!
//! The output has shape `(count, 2)`; its length implies the shape.
use crate::prelude::*;

const ROWS: usize = 2;
const COLS: usize = 2;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[i64; COLS]; ROWS] = input.read_matrix();
    let min = a.iter().flatten().copied().min().expect("a is not empty");
    let mut positions = Vec::new();
    for (i, row) in a.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v == min {
                positions.extend([i as i64, j as i64]);
            }
        }
    }
    positions
}
