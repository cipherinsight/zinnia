//! DS-1000 case 0438: Pearson correlation of two integer series,
//! `np.corrcoef(post, distance)[0][1]`.
//!
//! ```python
//! def kernel(post: Integer[Array, "4"], distance: Integer[Array, "4"]):
//!     result = np.corrcoef(post, distance)[0][1]
//!     return result
//! ```
use crate::prelude::*;

const N: usize = 4;

pub type Output = Q32;

pub fn main(input: &mut impl Input) -> Output {
    let post: [i64; N] = input.read_array();
    let distance: [i64; N] = input.read_array();
    pearson(&post.map(Q32::from_int), &distance.map(Q32::from_int))
}

/// `sum(dx * dy) / sqrt(sum(dx^2) * sum(dy^2))` with `dx = x - mean(x)` and
/// `dy = y - mean(y)`. NumPy's result is nan when a series is constant; that
/// division by zero panics here.
fn pearson(x: &[Q32; N], y: &[Q32; N]) -> Q32 {
    let (mean_x, mean_y) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (Q32::ZERO, Q32::ZERO, Q32::ZERO);
    for (&xi, &yi) in x.iter().zip(y) {
        let (dx, dy) = (xi - mean_x, yi - mean_y);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx * syy).sqrt()
}

fn mean(values: &[Q32]) -> Q32 {
    values.iter().sum::<Q32>() / values.len() as i64
}
