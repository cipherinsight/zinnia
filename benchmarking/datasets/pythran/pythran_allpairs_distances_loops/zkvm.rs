//! Pythran allpairs_distances_loops: squared distances between all rows of a
//! 500 x d and a 200 x d matrix of ones (so every entry is 0).
//!
//! ```python
//! def dists(X, Y):
//!     result = np.zeros((X.shape[0], Y.shape[0]), X.dtype)
//!     for i in range(X.shape[0]):
//!         for j in range(Y.shape[0]):
//!             result[i, j] = np.sum((X[i, :] - Y[j, :]) ** 2)
//!     return result
//!
//! def allpairs_distances_loops(d: Integer):
//!     X = np.ones((500, d))
//!     Y = np.ones((200, d))
//!     return dists(X, Y)
//! ```
use crate::prelude::*;

/// `dists`: the squared Euclidean distance between every row of `x` and
/// every row of `y`, each square rescaled and then summed.
fn dists(x: &Array2<Q32>, y: &Array2<Q32>) -> Array2<Q32> {
    Array2::from_shape_fn((x.nrows(), y.nrows()), |(i, j)| {
        x.row(i)
            .iter()
            .zip(y.row(j))
            .map(|(&a, &b)| {
                let t = a - b;
                t * t
            })
            .sum()
    })
}

pub mod allpairs_distances_loops {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let d: i64 = args.scalar();
        let d = usize::try_from(d).expect("negative dimensions are not allowed");
        // The 500 and 200 rows are constants of the kernel.
        let x = Array2::<Q32>::ones((500, d));
        let y = Array2::<Q32>::ones((200, d));
        dists(&x, &y)
    }
}
