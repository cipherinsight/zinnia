//! DS-1000 case 0320: (row, column) of the minimum.
//!
//!     def kernel(a: Float[Array, "5 6"]):
//!         result = np.unravel_index(a.argmin(), a.shape)
//!         return result
// output: (i64, i64)
use crate::np::argmin;
use crate::q32::Q32;

const ROWS: u32 = 5;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>) -> (i64, i64) {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let flat = argmin(a.span());
    ((flat / COLS).into(), (flat % COLS).into())
}
