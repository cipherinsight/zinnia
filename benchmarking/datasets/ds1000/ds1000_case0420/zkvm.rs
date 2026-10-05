//! DS-1000 case 0420: smooth clamp of `x` to `[x_min, x_max]`.
//!
//! ```python
//! def kernel(x: Float, x_min: Integer, x_max: Integer):
//!     return np.where(x < x_min, x_min, np.where(x > x_max, x_max, 3*x**2 - 2*x**3))
//! ```
//!
//! The result is a float (`np.where` promotes the integer bounds); only the
//! selected branch is computed.
use crate::prelude::*;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let x: Q32 = input.read();
    let x_min = Q32::from_int(input.read());
    let x_max = Q32::from_int(input.read());
    if x < x_min {
        x_min
    } else if x > x_max {
        x_max
    } else {
        smoothstep(x)
    }
}

/// `3*x**2 - 2*x**3`.
fn smoothstep(x: Q32) -> Q32 {
    let square = x * x;
    let cube = square * x;
    square * 3 - cube * 2
}
