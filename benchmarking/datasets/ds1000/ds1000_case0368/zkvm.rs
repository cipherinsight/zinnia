//! DS-1000 case 0368: are all rows of `a` close to its first row?
//!
//! ```python
//! def kernel(a: Float[Array, "3 3"]):
//!     result = np.isclose(a, a[0], atol=0).all()
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 3;

pub type Output = bool;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; N]; N] = input.read_matrix();
    let close = |x, y| np::isclose(x, y, np::DEFAULT_RTOL, Q32::ZERO);
    // Row 0 is trivially close to itself.
    a[1..].iter().all(|row| row.iter().zip(&a[0]).all(|(&x, &y)| close(x, y)))
}
