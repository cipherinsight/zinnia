//! DS-1000 case 0369: is every row of `a` close to its first entry?
//!
//! ```python
//! def kernel(a: Float[Array, "3 3"]):
//!     result =np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 3;

/// NumPy's default `rtol = 1e-5`, quantized: `round(1e-5 * 2^32)`.
const RTOL: Q32 = Q32::from_bits(42_950);

pub type Output = bool;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; N]; N] = input.read_matrix();
    // Column 0 is trivially close to itself.
    a.iter().all(|row| row[1..].iter().all(|&x| isclose(x, row[0])))
}

/// `np.isclose(x, y, atol=0)`: `|x - y| <= rtol * |y|`.
fn isclose(x: Q32, y: Q32) -> bool {
    (x - y).abs() <= RTOL * y.abs()
}
