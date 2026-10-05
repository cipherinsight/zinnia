//! DS-1000 case 0385: split `a` into 2x2 blocks, listed column of blocks by
//! column of blocks.
//!
//!     def kernel(a: Float[Array, "100 200"]):
//!         result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 100;
const COLS: u32 = 200;

/// The reshape and `swapaxes` index block `(i, j)` (rows `2i, 2i+1`, columns
/// `2j, 2j+1`) as `[i][j]`; the transpose makes that `[j][i]`, so the blocks
/// come out with `j` outer and `i` inner.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    for j in 0..COLS / 2 {
        for i in 0..ROWS / 2 {
            append_block(ref result, a, i, j);
        }
    }
    result
}

/// Appends the 2x2 block at rows `2i, 2i+1` and columns `2j, 2j+1`, row-major.
fn append_block(ref out: Array<Q32>, a: Span<Q32>, i: u32, j: u32) {
    let top = 2 * i * COLS + 2 * j;
    let bottom = top + COLS;
    out.append(*a.at(top));
    out.append(*a.at(top + 1));
    out.append(*a.at(bottom));
    out.append(*a.at(bottom + 1));
}
