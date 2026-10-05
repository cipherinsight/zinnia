//! DS-1000 case 0312: (row, column) of the maximum.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         result = np.unravel_index(a.argmax(), a.shape)
//!         return result
// output: (i64, i64)
use crate::np::argmax;
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>) -> (i64, i64) {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let flat = argmax(a.span());
    ((flat / COLS).into(), (flat % COLS).into())
}
