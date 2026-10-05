//! DS-1000 case 0439: the outer product of every column of `X` with itself,
//! `X.T[:, :, None] * X.T[:, None]`, of shape `(20, 10, 10)`:
//! `result[i][j][k] = X[j][i] * X[k][i]`.
//!
//!     def kernel(X: Float[Array, "10 20"]):
//!         result = X.T[:, :, None] * X.T[:, None]
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 10;
const COLS: u32 = 20;

#[executable]
fn main(X: Array<Q32>) -> Array<Q32> {
    assert!(X.len() == ROWS * COLS, "X has the wrong length");
    let mut out = array![];
    for i in 0..COLS {
        for j in 0..ROWS {
            let x_ji = *X.at(j * COLS + i);
            for k in 0..ROWS {
                out.append(x_ji * *X.at(k * COLS + i));
            }
        }
    }
    out
}
