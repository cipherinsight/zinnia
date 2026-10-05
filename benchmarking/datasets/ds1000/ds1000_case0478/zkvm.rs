//! DS-1000 case 0478: `a` with everything off the diagonal set to zero,
//! written through the diagonal view `np.einsum('ii->i', a)`.
//!
//! ```python
//! def kernel(a: Integer[Array, "4 4"]):
//!     result = np.einsum('ii->i', a)
//!     save = result.copy()
//!     a[...] = 0
//!     result[...] = save
//!     return a
//! ```
use crate::prelude::*;

const N: usize = 4;

pub type Output = Vec<i64>;

/// `np.einsum('ii->i', a)` is a writeable view of the diagonal of `a`:
/// `a[...] = 0` clears it with the rest of `a`, and `result[...] = save`
/// writes the saved diagonal back into `a`, which the kernel returns.
pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<i64> = input.read_vec(N * N);
    let mut result = vec![0; N * N];
    for i in 0..N {
        result[i * N + i] = a[i * N + i];
    }
    result
}
