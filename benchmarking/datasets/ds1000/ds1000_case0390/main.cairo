//! DS-1000 case 0390: cut `a` into `patch_size` x `patch_size` patches,
//! listed column of patches by column of patches.
//!
//!     def kernel(a: Float[Array, "100 200"]):
//!         patch_size = 5
//!         x = a[:a.shape[0] // patch_size * patch_size, :a.shape[1] // patch_size * patch_size]
//!         result = x.reshape(x.shape[0]//patch_size, patch_size, x.shape[1]// patch_size, patch_size).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, patch_size, patch_size)
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 100;
const COLS: u32 = 200;
const PATCH: u32 = 5;
/// The shape of `x`: the rows and columns that whole patches cover.
const CROP_ROWS: u32 = ROWS / PATCH * PATCH;
const CROP_COLS: u32 = COLS / PATCH * PATCH;

/// The reshape and `swapaxes` index patch `(i, j)` (rows `i * PATCH ..`,
/// columns `j * PATCH ..`) as `[i][j]`; the transpose makes that `[j][i]`, so
/// the patches come out with `j` outer and `i` inner. Rows and columns
/// beyond the last whole patch are cut off (none here: both sides are
/// multiples of `PATCH`).
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    for j in 0..CROP_COLS / PATCH {
        for i in 0..CROP_ROWS / PATCH {
            append_patch(ref result, a, i, j);
        }
    }
    result
}

/// Appends patch `(i, j)` of `a`, row-major.
fn append_patch(ref out: Array<Q32>, a: Span<Q32>, i: u32, j: u32) {
    for r in i * PATCH..(i + 1) * PATCH {
        out.append_span(a.slice(r * COLS + j * PATCH, PATCH));
    }
}
