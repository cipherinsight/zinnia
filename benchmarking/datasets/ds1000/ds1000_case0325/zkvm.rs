//! DS-1000 case 0325: 1 if `number` read as radians has a larger sine than
//! `number` read as degrees, else 0.
//!
//! ```python
//! def kernel(number: Integer):
//!     deg = np.sin(np.deg2rad(number))
//!     rad = np.sin(number)
//!     result = int(rad > deg)
//!     return result
//! ```
use crate::prelude::*;

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let number = Q32::from_int(input.read::<i64>());
    let deg = (number * Q32::DEG_TO_RAD).sin();
    let rad = number.sin();
    (rad > deg) as i64
}
