//! DS-1000 case 0362: delete the columns listed 1-based in `del_col`,
//! ignoring numbers past the last column.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"]):
//!     del_col = np.array([12, 14, 1, 9])
//!     mask = (del_col <= a.shape[1])
//!     del_col = del_col[mask] - 1
//!     result = np.delete(a, del_col, axis=1)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
/// `del_col` as the kernel sets it: 1-based column numbers.
const DEL_COL: [usize; 4] = [12, 14, 1, 9];

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let kept: Vec<usize> = (0..COLS).filter(|&j| !is_deleted(j)).collect();
    a.chunks_exact(COLS).flat_map(|row| kept.iter().map(|&j| row[j])).collect()
}

/// Whether `del_col[del_col <= COLS] - 1` lists the 0-based column `j`; the
/// mask only drops numbers that match no column.
fn is_deleted(j: usize) -> bool {
    DEL_COL.contains(&(j + 1))
}
