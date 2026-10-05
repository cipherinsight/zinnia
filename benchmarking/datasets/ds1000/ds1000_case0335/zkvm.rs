//! DS-1000 case 0335: element-wise maximum of three arrays.
//!
//! ```python
//! def kernel(a: Float[Array, "50"], b: Float[Array, "50"], c: Float[Array, "50"]):
//!     result = np.max([a, b, c], axis=0)
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 50;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let b: Vec<Q32> = input.read_vec(N);
    let c: Vec<Q32> = input.read_vec(N);
    (0..N).map(|i| a[i].max(b[i]).max(c[i])).collect()
}
