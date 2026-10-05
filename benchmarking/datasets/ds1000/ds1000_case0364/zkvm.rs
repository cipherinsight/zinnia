//! DS-1000 case 0364: insert a row into a matrix,
//! `np.insert(a, pos, element, axis=0)`.
//!
//! ```python
//! def kernel(a: Float[Array, "100 10"], pos: Integer, element: Float[Array, "10"]):
//!     a = np.insert(a, pos, element, axis = 0)
//!     return a
//! ```
use crate::prelude::*;

const ROWS: usize = 100;
const COLS: usize = 10;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(ROWS * COLS);
    let pos: i64 = input.read();
    let element: Vec<Q32> = input.read_vec(COLS);
    let at = np::insert_position(pos, ROWS) * COLS;
    let mut result = Vec::with_capacity((ROWS + 1) * COLS);
    result.extend_from_slice(&a[..at]);
    result.extend_from_slice(&element);
    result.extend_from_slice(&a[at..]);
    result
}
