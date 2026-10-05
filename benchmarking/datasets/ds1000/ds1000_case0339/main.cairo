//! DS-1000 case 0339: the diagonal and the anti-diagonal of the leading
//! square block of `a`, stacked as two rows.
//!
//!     def kernel(a: Float[Array, "8 10"]):
//!         dim = min(a.shape)
//!         b = a[:dim,:dim]
//!         result = np.vstack((np.diag(b), np.diag(np.fliplr(b))))
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 10;
/// `min(a.shape)`.
const DIM: u32 = if ROWS < COLS {
    ROWS
} else {
    COLS
};

/// `b[i][j]` is `a[i][j]`, so the anti-diagonal of `b` is `a[i][DIM - 1 - i]`.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut result = array![];
    for i in 0..DIM {
        result.append(*a.at(i * COLS + i));
    }
    for i in 0..DIM {
        result.append(*a.at(i * COLS + (DIM - 1 - i)));
    }
    result
}
