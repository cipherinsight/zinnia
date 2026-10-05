//! DS-1000 case 0382: the indices that sort `a`, `np.argsort(a)`.
//!
//! ```python
//! def kernel(a: Float[Array, "100"]):
//!     result = np.argsort(a)
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 100;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let mut order: Vec<usize> = (0..N).collect();
    // A stable sort: ties stay in index order.
    order.sort_by_key(|&i| a[i]);
    order.into_iter().map(|i| i as i64).collect()
}
