//! DS-1000 case 0357: reorder `b` along the first axis by the order that sorts
//! `a` along that axis in decreasing order (the argsort reversed).
//!
//! ```python
//! def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
//!     sort_indices = np.argsort(a, axis=0)[::-1, :, :]
//!     static_indices = np.indices(a.shape)
//!     c = b[sort_indices, static_indices[1], static_indices[2]]
//!     return c
//! ```
use crate::prelude::*;

/// Length of the sorted first axis.
const DEPTH: usize = 3;
/// Number of positions on the last two axes (3 x 3), flattened.
const PLANE: usize = 3 * 3;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [[Q32; PLANE]; DEPTH] = input.read_matrix();
    let b: [[i64; PLANE]; DEPTH] = input.read_matrix();
    let orders = np::argsort_axis0(&a);
    let mut c = Vec::with_capacity(DEPTH * PLANE);
    for i in (0..DEPTH).rev() {
        for (pos, order) in orders.iter().enumerate() {
            c.push(b[order[i]][pos]);
        }
    }
    c
}
