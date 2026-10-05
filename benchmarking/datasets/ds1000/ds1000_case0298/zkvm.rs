//! DS-1000 case 0298: one-hot encoding of `a` over its sorted distinct values.
//!
//! ```python
//! def kernel(a: Float[Array, "5"]):
//!     vals, idx = np.unique(a, return_inverse=True)
//!     b = np.zeros((a.size, vals.size))
//!     b[np.arange(a.size), idx] = 1
//!     return b
//! ```
//!
//! The output has shape `(5, vals.size)`; its length implies the shape.
use crate::prelude::*;

const N: usize = 5;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [Q32; N] = input.read_array();
    let vals = unique(&a);
    let mut b = vec![Q32::ZERO; N * vals.len()];
    for (i, v) in a.iter().enumerate() {
        let column = vals.binary_search(v).expect("every value of a is in vals");
        b[i * vals.len() + column] = Q32::ONE;
    }
    b
}

/// `np.unique`: the distinct values in increasing order.
fn unique(values: &[Q32]) -> Vec<Q32> {
    let mut vals = values.to_vec();
    vals.sort_unstable();
    vals.dedup();
    vals
}
