//! DS-1000 case 0360: drop the third row, `np.delete(a, 2, axis=0)`.
//!
//!     def kernel(a: Float[Array, "4 4"]):
//!         a = np.delete(a, 2, axis = 0)
//!         return a
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 4;
const COLS: u32 = 4;
const DELETED: u32 = 2;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    result.append_span(a.slice(0, DELETED * COLS));
    result.append_span(a.slice((DELETED + 1) * COLS, (ROWS - DELETED - 1) * COLS));
    result
}
