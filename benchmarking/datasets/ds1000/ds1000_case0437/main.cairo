//! DS-1000 case 0437: mask of the entries equal to their row's minimum.
//!
//!     def kernel(a: Float[Array, "10 5"]):
//!         mask = (a.min(axis=1,keepdims=1) == a)
//!         return mask
// output: [bool]
use crate::q32::Q32;

const ROWS: u32 = 10;
const COLS: u32 = 5;

#[executable]
fn main(a: Array<Q32>) -> Array<bool> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut mask = array![];
    for r in 0..ROWS {
        let row = a.span().slice(r * COLS, COLS);
        let mut min = *row.at(0);
        for v in row {
            if *v < min {
                min = *v;
            }
        }
        for v in row {
            mask.append(*v == min);
        }
    }
    mask
}
