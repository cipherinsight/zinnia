//! DS-1000 case 0326: arcsine in degrees.
//!
//! ```python
//! def kernel(value: Float):
//!     result = np.degrees(np.arcsin(value))
//!     return result
//! ```
use crate::prelude::*;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let value: Q32 = input.read();
    value.asin() * Q32::RAD_TO_DEG
}
