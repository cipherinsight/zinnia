//! DS-1000 case 0338: the diagonal and the anti-diagonal of a square matrix,
//! stacked as two rows.
//!
//! ```python
//! def kernel(a: Float[Array, "8 8"]):
//!     result = np.vstack((np.diag(a), np.diag(np.fliplr(a))))
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 8;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N * N);
    let diagonal = (0..N).map(|i| a[i * N + i]);
    let anti_diagonal = (0..N).map(|i| a[i * N + (N - 1 - i)]);
    diagonal.chain(anti_diagonal).collect()
}
