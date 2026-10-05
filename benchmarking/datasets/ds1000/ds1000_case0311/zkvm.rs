//! DS-1000 case 0311: position of the maximum, its flat (C-order) index
//! unravelled in Fortran order.
//!
//! ```python
//! def kernel(a: Float[Array, "8 6"]):
//!     result = np.unravel_index(a.argmax(), a.shape, order = 'F')
//!     return result
//! ```
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub type Output = (i64, i64);

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let flat = np::argmax(&a);
    // order='F': the first axis varies fastest.
    ((flat % ROWS) as i64, (flat / ROWS) as i64)
}
