//! DS-1000 case 0497: a matrix padded with the integer `element` at the
//! bottom and right to `shape = (93, 13)`.
//!
//!     def kernel(a: Float[Array, "42 10"], element: Integer):
//!         shape = (93, 13)
//!         result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant', constant_values=element)
//!         return result
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const ROWS: u32 = 42;
const COLS: u32 = 10;
const OUT_ROWS: u32 = 93;
const OUT_COLS: u32 = 13;

/// Each row of `a` followed by `element` to the full width, then rows of
/// `element`, converted to the float dtype of `a`.
#[executable]
fn main(a: Array<Q32>, element: i64) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let fill = Q32Trait::from_int(element);
    let mut result = array![];
    for r in 0..ROWS {
        result.append_span(a.slice(r * COLS, COLS));
        append_repeated(ref result, fill, OUT_COLS - COLS);
    }
    append_repeated(ref result, fill, (OUT_ROWS - ROWS) * OUT_COLS);
    result
}

fn append_repeated(ref out: Array<Q32>, value: Q32, count: u32) {
    for _ in 0..count {
        out.append(value);
    }
}
