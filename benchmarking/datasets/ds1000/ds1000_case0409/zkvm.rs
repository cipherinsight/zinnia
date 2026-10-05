//! DS-1000 case 0409: element-wise sum of two matrices.
//!
//! ```python
//! def kernel(x: Float[Array, "8 6"], y: Float[Array, "8 6"]):
//!     x_new = np.array(x)
//!     y_new = np.array(y)
//!     z = x_new + y_new
//!     return z
//! ```
use crate::prelude::*;

const N: usize = 8 * 6;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let x: Vec<Q32> = input.read_vec(N);
    let y: Vec<Q32> = input.read_vec(N);
    x.iter().zip(&y).map(|(&xi, &yi)| xi + yi).collect()
}
