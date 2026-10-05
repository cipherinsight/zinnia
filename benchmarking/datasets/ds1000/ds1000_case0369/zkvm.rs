//! DS-1000 case 0369: is every row of `a` close to its first entry?
//!
//! ```python
//! def kernel(a: Float[Array, "3 3"]):
//!     result =np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 3;

pub type Output = bool;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; N]; N] = input.read_matrix();
    let close = |x, y| np::isclose(x, y, np::DEFAULT_RTOL, Q32::ZERO);
    // Column 0 is trivially close to itself.
    a.iter().all(|row| row[1..].iter().all(|&x| close(x, row[0])))
}
