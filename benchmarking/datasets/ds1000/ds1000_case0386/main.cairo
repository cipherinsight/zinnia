//! DS-1000 case 0386: overlapping 2x2 windows, reshaped to (-1, 2, 2).
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 100;
const COLS: u32 = 200;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    // Preserve [window row, window column, row, column] order on reshaping.
    for i in 0..ROWS - 1 {
        for j in 0..COLS - 1 {
            let top = i * COLS + j;
            result.append(*a.at(top));
            result.append(*a.at(top + 1));
            result.append(*a.at(top + COLS));
            result.append(*a.at(top + COLS + 1));
        }
    }
    result
}
