//! DS-1000 case 0440: the square roots of the diagonals of the matrices in
//! `Y`, one matrix per column.
//!
//!     def kernel(Y: Float[Array, "10 5 5"]):
//!         X = np.zeros([Y.shape[1], Y.shape[0]])
//!         for i, mat in enumerate(Y):
//!             diag = np.sqrt(np.diag(mat))
//!             X[:, i] += diag
//!         return X
// output: [q32]
use crate::q32::{Q32, Q32Trait};

const MATRICES: u32 = 10;
const SIZE: u32 = 5;

/// Returns `X`, shape `(5, 10)`, flattened row-major.
#[executable]
fn main(y: Array<Q32>) -> Array<Q32> {
    assert!(y.len() == MATRICES * SIZE * SIZE, "Y has the wrong length");
    // `X[j, i] = sqrt(Y[i, j, j])`: `X` starts at zero, so `+=` assigns.
    let mut x = array![];
    for j in 0..SIZE {
        for i in 0..MATRICES {
            x.append(y.at((i * SIZE + j) * SIZE + j).sqrt());
        }
    }
    x
}
