//! Pythran morphology: grey-scale dilation with a `k x k` square window,
//! decomposed into a vertical then a horizontal running maximum (windows
//! clipped at the borders).
//!
//! ```python
//! @requires(lambda x, k: 0 <= k <= 1024)
//! def dilate_decompose_loops(x: Float[Array, '100 200'], k: Integer):
//!     m, n = x.shape
//!     y = empty_like(x)
//!     for i in range(m):
//!         for j in range(n):
//!             left_idx = max(0, i - k // 2)
//!             right_idx = min(m, i + k // 2 + 1)
//!             currmax = x[left_idx, j]
//!             for ii in range(left_idx + 1, right_idx):
//!                 elt = x[ii, j]
//!                 if elt > currmax:
//!                     currmax = elt
//!             y[i, j] = currmax
//!     z = empty_like(x)
//!     for i in range(m):
//!         for j in range(n):
//!             left_idx = max(0, j - k // 2)
//!             right_idx = min(n, j + k // 2 + 1)
//!             currmax = y[i, left_idx]
//!             for jj in range(left_idx + 1, right_idx):
//!                 elt = y[i, jj]
//!                 if elt > currmax:
//!                     currmax = elt
//!             z[i, j] = currmax
//!     return z
//! ```
use crate::prelude::*;

/// The maximum of `values[max(0, c - half) .. min(len, c + half + 1)]`,
/// scanned from the left (the window always holds `values[c]`).
fn window_max(values: ArrayView1<Q32>, c: usize, half: usize) -> Q32 {
    let left = c.saturating_sub(half);
    let right = values.len().min(c + half + 1);
    let mut currmax = values[left];
    for &elt in values.slice(s![left + 1..right]) {
        if elt > currmax {
            currmax = elt;
        }
    }
    currmax
}

pub mod dilate_decompose_loops {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array2<Q32> = args.array2();
        let k: i64 = args.scalar();
        assert!((0..=1024).contains(&k), "requires 0 <= k <= 1024");
        let half = (k / 2) as usize;
        let y = Array2::from_shape_fn(x.dim(), |(i, j)| window_max(x.column(j), i, half));
        Array2::from_shape_fn(x.dim(), |(i, j)| window_max(y.row(i), j, half))
    }
}
