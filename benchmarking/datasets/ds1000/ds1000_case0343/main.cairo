//! DS-1000 case 0343: the elements of `X` in column-major order (`X.T.flat`).
//!
//!     def kernel(X: Integer[Array, "5 6"]):
//!         result = []
//!         for value in X.T.flat:
//!             result.append(value)
//!         return result
// output: [i64]

const ROWS: u32 = 5;
const COLS: u32 = 6;

/// Row `j` of `X.T` is column `j` of `X`.
#[executable]
fn main(X: Array<i64>) -> Array<i64> {
    assert!(X.len() == ROWS * COLS, "X has the wrong length");
    let mut result = array![];
    for j in 0..COLS {
        for i in 0..ROWS {
            result.append(*X.at(i * COLS + j));
        }
    }
    result
}
