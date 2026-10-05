//! DS-1000 case 0390: cut `a` into `patch_size` x `patch_size` patches,
//! listed column of patches by column of patches.
//!
//! ```python
//! def kernel(a: Float[Array, "100 200"]):
//!     patch_size = 5
//!     x = a[:a.shape[0] // patch_size * patch_size, :a.shape[1] // patch_size * patch_size]
//!     result = x.reshape(x.shape[0]//patch_size, patch_size, x.shape[1]// patch_size, patch_size).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, patch_size, patch_size)
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 100;
const COLS: usize = 200;
const PATCH: usize = 5;
/// The shape of `x`: the rows and columns that whole patches cover.
const CROP_ROWS: usize = ROWS / PATCH * PATCH;
const CROP_COLS: usize = COLS / PATCH * PATCH;

pub type Output = Vec<Q32>;

/// The reshape and `swapaxes` index patch `(i, j)` (rows `i * PATCH ..`,
/// columns `j * PATCH ..`) as `[i][j]`; the transpose makes that `[j][i]`, so
/// the patches come out with `j` outer and `i` inner. Rows and columns
/// beyond the last whole patch are cut off (none here: both sides are
/// multiples of `PATCH`).
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let mut result = Vec::with_capacity(CROP_ROWS * CROP_COLS);
    for j in 0..CROP_COLS / PATCH {
        for i in 0..CROP_ROWS / PATCH {
            push_patch(&mut result, &a, i, j);
        }
    }
    result
}

/// Appends patch `(i, j)` of `a`, row-major.
fn push_patch(out: &mut Vec<Q32>, a: &[Q32], i: usize, j: usize) {
    for r in i * PATCH..(i + 1) * PATCH {
        let start = r * COLS + j * PATCH;
        out.extend_from_slice(&a[start..start + PATCH]);
    }
}
