//! DS-1000 case 0453: every row of an integer matrix divided by its L2 norm.
//!
//!     def kernel(X: Integer[Array, "5 4"]):
//!         l2 = np.sqrt((X*X).sum(axis=-1))
//!         result = X / l2.reshape(-1, 1)
//!         return result
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const ROWS: u32 = 5;
const COLS: u32 = 4;

/// `X*X` and its row sums are integers; `np.sqrt` turns the sum into a float.
/// NumPy gives nan for a row of zeros; that division by zero fails here.
#[executable]
fn main(X: Array<i64>) -> Array<Q32> {
    assert!(X.len() == ROWS * COLS, "X has the wrong length");
    let mut out = array![];
    for r in 0..ROWS {
        let row = X.span().slice(r * COLS, COLS);
        let mut squares: i64 = 0;
        for v in row {
            squares += *v * *v;
        }
        let l2 = Q32Trait::from_int(squares).sqrt();
        for v in row {
            out.append(Q32Trait::from_int(*v) / l2);
        }
    }
    out
}
