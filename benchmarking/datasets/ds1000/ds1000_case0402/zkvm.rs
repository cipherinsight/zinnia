//! DS-1000 case 0402: the Fortran-order linear index of `index` in an array
//! of shape `dims`.
//!
//! ```python
//! def kernel(a: Float[Array, "8 9 8 8 8"], index: Integer[Array, "5"]):
//!     dims = np.array([8, 9, 8, 8, 8])
//!     result = np.ravel_multi_index(index, dims=dims, order='F')
//!     return result
//! ```
// unused parameters: a
use crate::prelude::*;

const DIMS: [i64; 5] = [8, 9, 8, 8, 8];

pub type Output = i64;

pub fn main(input: &mut impl Input) -> Output {
    let index: [i64; 5] = input.read_array();
    // Fortran order: the first index varies fastest, so Horner's rule runs
    // from the last axis to the first.
    let mut flat = 0;
    for (&i, &dim) in index.iter().zip(&DIMS).rev() {
        assert!((0..dim).contains(&i), "invalid entry in coordinates array");
        flat = flat * dim + i;
    }
    flat
}
