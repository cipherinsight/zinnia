//! DS-1000 case 0387: split `a` into 2x2 blocks, listed row of blocks by row
//! of blocks.
//!
//! ```python
//! def kernel(a: Float[Array, "100 200"]):
//!     result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).swapaxes(1, 2).reshape(-1, 2, 2)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 100;
const COLS: usize = 200;

pub type Output = Vec<Q32>;

/// The reshape and `swapaxes` index block `(i, j)` (rows `2i, 2i+1`, columns
/// `2j, 2j+1`) as `[i][j]`, so the blocks come out in row-major block order.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let mut result = Vec::with_capacity(ROWS * COLS);
    for i in 0..ROWS / 2 {
        for j in 0..COLS / 2 {
            push_block(&mut result, &a, i, j);
        }
    }
    result
}

/// Appends the 2x2 block at rows `2i, 2i+1` and columns `2j, 2j+1`, row-major.
fn push_block(out: &mut Vec<Q32>, a: &[Q32], i: usize, j: usize) {
    let top = 2 * i * COLS + 2 * j;
    out.extend_from_slice(&a[top..top + 2]);
    out.extend_from_slice(&a[top + COLS..top + COLS + 2]);
}
