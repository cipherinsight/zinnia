//! Pythran allpairs: squared Euclidean distances between the rows of two
//! matrices, as a nested list comprehension and as explicit loops.
//!
//! ```python
//! def sqr_dists(X: Float[Array, '300 300'], Y: Float[Array, '300 300']):
//!     return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])
//!
//! def sqr_dists_loops(X: Float[Array, '8 8'], Y: Float[Array, '8 8']):
//!     m, n = (X.shape[0], Y.shape[0])
//!     D = np.zeros((m, n))
//!     for i in range(m):
//!         for j in range(n):
//!             D[i, j] = np.sum((X[i] - Y[j]) ** 2)
//!     return D
//! ```
use crate::prelude::*;

/// `np.sum((x - y) ** 2)`: every square rescaled, then summed.
fn sqr_dist(x: ArrayView1<Q32>, y: ArrayView1<Q32>) -> Q32 {
    assert_eq!(x.len(), y.len(), "operands could not be broadcast together");
    x.iter()
        .zip(&y)
        .map(|(&a, &b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

pub mod sqr_dists {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array2<Q32> = args.array2();
        let y: Array2<Q32> = args.array2();
        // The outer comprehension runs over Y: row j of the result is y = Y[j].
        Array2::from_shape_fn((y.nrows(), x.nrows()), |(j, i)| {
            sqr_dist(x.row(i), y.row(j))
        })
    }
}

pub mod sqr_dists_loops {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array2<Q32> = args.array2();
        let y: Array2<Q32> = args.array2();
        Array2::from_shape_fn((x.nrows(), y.nrows()), |(i, j)| {
            sqr_dist(x.row(i), y.row(j))
        })
    }
}
