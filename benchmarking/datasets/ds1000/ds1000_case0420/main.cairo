//! DS-1000 case 0420: smooth clamp of `x` to `[x_min, x_max]`.
//!
//!     def kernel(x: Float, x_min: Integer, x_max: Integer):
//!         return np.where(x < x_min, x_min, np.where(x > x_max, x_max, 3*x**2 - 2*x**3))
//!
//! The result is a float (`np.where` promotes the integer bounds); only the
//! selected branch is computed.
// output: q32
use crate::q32::{Q32, Q32Trait};

#[executable]
fn main(x: Q32, x_min: i64, x_max: i64) -> Q32 {
    let x_min = Q32Trait::from_int(x_min);
    let x_max = Q32Trait::from_int(x_max);
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
    square.mul_int(3) - cube.mul_int(2)
}
