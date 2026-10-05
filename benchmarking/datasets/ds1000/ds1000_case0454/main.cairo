//! DS-1000 case 0454: every row of an integer matrix divided by its L-infinity
//! norm.
//!
//!     def kernel(X: Integer[Array, "5 4"]):
//!         linf = np.abs(X).max(axis = 1)
//!         result = X / linf.reshape(-1, 1)
//!         return result
// output: [q32]
use crate::np::abs;
use crate::q32::{Q32, Q32Trait};

const ROWS: u32 = 5;
const COLS: u32 = 4;

/// The norm is an integer and `/` is NumPy's true division, so each entry is
/// `float(x) / float(linf)`. NumPy gives nan for a row of zeros; that division
/// by zero fails here.
#[executable]
fn main(X: Array<i64>) -> Array<Q32> {
    assert!(X.len() == ROWS * COLS, "X has the wrong length");
    let mut out = array![];
    for r in 0..ROWS {
        let row = X.span().slice(r * COLS, COLS);
        let mut linf: i64 = 0;
        for v in row {
            let magnitude = abs(*v);
            if magnitude > linf {
                linf = magnitude;
            }
        }
        for v in row {
            out.append(Q32Trait::from_int(*v).div_int(linf));
        }
    }
    out
}
