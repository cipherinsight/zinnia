//! DS-1000 case 0324: cosine of an angle given in degrees.
//!
//! ```python
//! def kernel(degree: Integer):
//!     result = np.cos(np.deg2rad(degree))
//!     return result
//! ```
use crate::prelude::*;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let degree: i64 = input.read();
    (Q32::from_int(degree) * Q32::DEG_TO_RAD).cos()
}
