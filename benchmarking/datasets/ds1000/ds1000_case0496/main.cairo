//! DS-1000 case 0496: a matrix padded with zeros at the bottom and right to
//! `shape = (93, 13)`.
//!
//!     def kernel(a: Float[Array, "42 10"]):
//!         shape = (93, 13)
//!         result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant')
//!         return result
// output: [q32]
use crate::np::append_repeated;
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 42;
const COLS: u32 = 10;
const OUT_ROWS: u32 = 93;
const OUT_COLS: u32 = 13;

/// Each row of `a` followed by zeros to the full width, then rows of zeros.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    for r in 0..ROWS {
        result.append_span(a.slice(r * COLS, COLS));
        append_repeated(ref result, ZERO, OUT_COLS - COLS);
    }
    append_repeated(ref result, ZERO, (OUT_ROWS - ROWS) * OUT_COLS);
    result
}
