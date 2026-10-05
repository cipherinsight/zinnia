//! DS-1000 case 0499: a matrix centred in zeros of `shape = (93, 13)`; an odd
//! padding puts the extra row or column at the bottom or right.
//!
//!     @zk_chip
//!     def to_shape(a, shape):
//!         y_, x_ = shape
//!         y, x = a.shape
//!         y_pad = (y_-y)
//!         x_pad = (x_-x)
//!         return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
//!                             (x_pad//2, x_pad//2 + x_pad%2)),
//!                         mode = 'constant')
//!
//!     @zk_circuit
//!     def kernel(a: Float[Array, "42 10"]):
//!         shape = (93, 13)
//!         result = to_shape(a, shape)
//!         return result
// output: [q32]
use crate::q32::{Q32, ZERO};

const ROWS: u32 = 42;
const COLS: u32 = 10;
const OUT_ROWS: u32 = 93;
const OUT_COLS: u32 = 13;
/// `y_pad // 2`: zero rows above `a`.
const TOP: u32 = (OUT_ROWS - ROWS) / 2;
/// `x_pad // 2`: zero columns left of `a`.
const LEFT: u32 = (OUT_COLS - COLS) / 2;

/// `TOP` zero rows; each row of `a` between `LEFT` zeros and the zeros up to
/// the full width; zero rows up to the full height.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let a = a.span();
    let mut result = array![];
    append_zeros(ref result, TOP * OUT_COLS);
    for r in 0..ROWS {
        append_zeros(ref result, LEFT);
        result.append_span(a.slice(r * COLS, COLS));
        append_zeros(ref result, OUT_COLS - LEFT - COLS);
    }
    append_zeros(ref result, (OUT_ROWS - TOP - ROWS) * OUT_COLS);
    result
}

fn append_zeros(ref out: Array<Q32>, count: u32) {
    for _ in 0..count {
        out.append(ZERO);
    }
}
