//! DS-1000 case 0329: element-wise sixth power, `a ** power` with `power = 6`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"]):
//!     power = 6
//!     a = a ** power
//!     return a
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
const POWER: u32 = 6;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    a.into_iter().map(|v| pow(v, POWER)).collect()
}

/// `x ** exp` for a non-negative integer exponent by square-and-multiply
/// (for 6: `x^2`, `x^4`, `x^2 * x^4`); every product rescales.
fn pow(mut base: Q32, mut exp: u32) -> Q32 {
    let mut result = Q32::ONE;
    loop {
        if exp & 1 == 1 {
            result *= base;
        }
        exp >>= 1;
        if exp == 0 {
            return result;
        }
        base *= base;
    }
}
