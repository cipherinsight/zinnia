//! DS-1000 case 0329: element-wise sixth power, `a ** power` with `power = 6`.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         power = 6
//!         a = a ** power
//!         return a
// output: [q32]
use crate::q32::{ONE, Q32};

const ROWS: u32 = 8;
const COLS: u32 = 6;
const POWER: u32 = 6;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let mut result = array![];
    for v in a {
        result.append(pow(v, POWER));
    }
    result
}

/// `x ** exp` for a non-negative integer exponent by square-and-multiply
/// (for 6: `x^2`, `x^4`, `x^2 * x^4`); every product rescales.
fn pow(x: Q32, exp: u32) -> Q32 {
    let mut base = x;
    let mut exp = exp;
    let mut result = ONE;
    loop {
        if exp % 2 == 1 {
            result = result * base;
        }
        exp /= 2;
        if exp == 0 {
            break;
        }
        base = base * base;
    }
    result
}
