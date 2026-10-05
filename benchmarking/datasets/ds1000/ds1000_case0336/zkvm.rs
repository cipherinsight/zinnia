//! DS-1000 case 0336: the anti-diagonal of a square matrix,
//! `np.diag(np.fliplr(a))`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 8"]):
//!     result = np.diag(np.fliplr(a))
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 8;

pub type Output = Vec<Q32>;

/// `np.fliplr` reverses every row, so entry `i` of its diagonal is
/// `a[i][N - 1 - i]`.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N * N);
    (0..N).map(|i| a[i * N + (N - 1 - i)]).collect()
}
