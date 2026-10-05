//! DS-1000 case 0485: `np.tensordot(A, B, axes=((2), (0)))`, i.e. the last
//! axis of `A` contracted with the first axis of `B`.
//!
//! ```python
//! def kernel(A: Float[Array, "5 6 3"], B: Float[Array, "3 3"]):
//!     result = np.tensordot(A,B,axes=((2),(0)))
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 5 * 6;
const K: usize = 3;
const COLS: usize = 3;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * K);
    let b: [[Q32; COLS]; K] = input.read_matrix();
    let mut out = Vec::with_capacity(ROWS * COLS);
    for row in a.chunks_exact(K) {
        for j in 0..COLS {
            out.push(dot((0..K).map(|k| (row[k], b[k][j]))));
        }
    }
    out
}
