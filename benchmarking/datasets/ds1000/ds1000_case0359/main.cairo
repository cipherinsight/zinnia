//! DS-1000 case 0359: drop the third column, `np.delete(a, 2, axis=1)`.
//!
//!     def kernel(a: Float[Array, "3 3"]):
//!         a = np.delete(a, 2, axis = 1)
//!         return a
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 3;
const COLS: u32 = 3;
const DELETED: u32 = 2;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    for r in 0..ROWS {
        let row = a.slice(r * COLS, COLS);
        result.append_span(row.slice(0, DELETED));
        result.append_span(row.slice(DELETED + 1, COLS - DELETED - 1));
    }
    result
}
