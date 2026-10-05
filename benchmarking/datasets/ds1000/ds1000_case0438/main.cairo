//! DS-1000 case 0438: Pearson correlation of two integer series,
//! `np.corrcoef(post, distance)[0][1]`.
//!
//!     def kernel(post: Integer[Array, "4"], distance: Integer[Array, "4"]):
//!         result = np.corrcoef(post, distance)[0][1]
//!         return result
// output: q32
use crate::q32::{Q32, Q32Trait, ZERO};

const N: u32 = 4;

/// `sum(dx * dy) / sqrt(sum(dx^2) * sum(dy^2))` with `dx = x - mean(x)` and
/// `dy = y - mean(y)`. NumPy's result is nan when a series is constant; that
/// division by zero fails here.
#[executable]
fn main(post: Array<i64>, distance: Array<i64>) -> Q32 {
    assert!(post.len() == N && distance.len() == N, "inputs have the wrong length");
    let mean_x = mean(post.span());
    let mean_y = mean(distance.span());
    let (mut sxy, mut sxx, mut syy) = (ZERO, ZERO, ZERO);
    for i in 0..N {
        let dx = Q32Trait::from_int(*post.at(i)) - mean_x;
        let dy = Q32Trait::from_int(*distance.at(i)) - mean_y;
        sxy = sxy + dx * dy;
        sxx = sxx + dx * dx;
        syy = syy + dy * dy;
    }
    sxy / (sxx * syy).sqrt()
}

/// The mean of integers, as a float.
fn mean(values: Span<i64>) -> Q32 {
    let mut sum: i64 = 0;
    for v in values {
        sum += *v;
    }
    Q32Trait::from_int(sum).div_int(values.len().into())
}
