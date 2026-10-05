//! DS-1000 case 0356: reorder `b` along the first axis by the order that sorts
//! `a` along that axis, `c[i, j, k] = b[np.argsort(a, axis=0)[i, j, k], j, k]`.
//!
//! ```python
//! def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
//!     sort_indices = np.argsort(a, axis=0)
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
    let a: Vec<Q32> = input.read_vec(DEPTH * PLANE);
    let b: Vec<i64> = input.read_vec(DEPTH * PLANE);
    let orders = argsort_axis0(&a);
    let mut c = Vec::with_capacity(DEPTH * PLANE);
    for i in 0..DEPTH {
        for (pos, order) in orders.iter().enumerate() {
            c.push(b[order[i] * PLANE + pos]);
        }
    }
    c
}

/// `np.argsort(a, axis=0)`: for every position of the last two axes, the
/// first-axis indices that sort `a` there (stable, ties in index order).
fn argsort_axis0(a: &[Q32]) -> [[usize; DEPTH]; PLANE] {
    core::array::from_fn(|pos| {
        let mut order: [usize; DEPTH] = core::array::from_fn(|i| i);
        order.sort_by_key(|&i| a[i * PLANE + pos]);
        order
    })
}
