//! DS-1000 case 0422: circular cross-correlation via
//! `np.correlate(a, np.hstack((b[1:], b)), mode='valid')`.
//!
//! ```python
//! def kernel(a: Float[Array, "50"], b: Float[Array, "50"]):
//!     result = np.correlate(a, np.hstack((b[1:], b)), mode='valid')
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 50;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(N);
    let b: Vec<Q32> = input.read_vec(N);
    let v: Vec<Q32> = b[1..].iter().chain(&b).copied().collect();
    correlate_valid_longer_v(&a, &v)
}

/// `np.correlate(a, v, mode='valid')` for real `v` longer than `a`. NumPy
/// then slides `a` along `v` and reverses the result:
/// `c[k] = sum_n a[n] * v[n + len(v) - len(a) - k]`, i.e. the windows of `v`
/// from the last to the first, each a fused dot product with `a`.
fn correlate_valid_longer_v(a: &[Q32], v: &[Q32]) -> Vec<Q32> {
    v.windows(a.len())
        .rev()
        .map(|window| dot(a.iter().copied().zip(window.iter().copied())))
        .collect()
}
