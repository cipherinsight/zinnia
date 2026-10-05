//! DS-1000 case 0440: the square roots of the diagonals of the matrices in
//! `Y`, one matrix per column.
//!
//! ```python
//! def kernel(Y: Float[Array, "10 5 5"]):
//!     X = np.zeros([Y.shape[1], Y.shape[0]])
//!     for i, mat in enumerate(Y):
//!         diag = np.sqrt(np.diag(mat))
//!         X[:, i] += diag
//!     return X
//! ```
use crate::prelude::*;

const MATRICES: usize = 10;
const SIZE: usize = 5;

/// `X`, shape `(5, 10)`, flattened row-major.
pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let y: Vec<Q32> = input.read_vec(MATRICES * SIZE * SIZE);
    // `X[j, i] = sqrt(Y[i, j, j])`: `X` starts at zero, so `+=` assigns.
    let mut x = Vec::with_capacity(SIZE * MATRICES);
    for j in 0..SIZE {
        for i in 0..MATRICES {
            x.push(y[(i * SIZE + j) * SIZE + j].sqrt());
        }
    }
    x
}
