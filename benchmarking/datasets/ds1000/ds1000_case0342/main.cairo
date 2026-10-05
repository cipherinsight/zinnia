//! DS-1000 case 0342: the elements of `X` in row-major order (`X.flat`).
//!
//!     def kernel(X: Integer[Array, "5 6"]):
//!         result = []
//!         for value in X.flat:
//!             result.append(value)
//!
//!
//!         return result
// output: [i64]

const ROWS: u32 = 5;
const COLS: u32 = 6;

/// `X.flat` visits the elements in row-major order, the order the input
/// holds them in, so the list is `X` itself.
#[executable]
fn main(X: Array<i64>) -> Array<i64> {
    assert!(X.len() == ROWS * COLS, "X has the wrong length");
    X
}
