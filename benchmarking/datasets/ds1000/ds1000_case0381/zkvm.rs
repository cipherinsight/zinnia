//! DS-1000 case 0381: the indices that sort `a` in decreasing order (the
//! argsort reversed; the final `[:len(a)]` keeps all of it).
//!
//! ```python
//! def kernel(a: Float[Array, "100"]):
//!     result = np.argsort(a)[::-1][:len(a)]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 100;

pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by_key(|&i| a[i]);
    order.iter().rev().map(|&i| i as i64).collect()
}
