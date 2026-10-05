//! Pythran euclidean_distance_square: the squared Euclidean distance between
//! every row of `x1` and every row of `x2`, through the Gram matrix.
//!
//! ```python
//! def euclidean_distance_square(x1: Float[Array, '1 3'], x2: Float[Array, '5 3']):
//!     return -2 * np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, np.newaxis] + np.sum(np.square(x2), axis=1)
//! ```
use crate::prelude::*;

/// `np.sum(np.square(x), axis=1)`: each square rescaled, then summed.
fn row_square_sums(x: &Array2<Q32>) -> Vec<Q32> {
    x.rows().into_iter().map(|row| row.iter().map(|&v| v * v).sum()).collect()
}

pub mod euclidean_distance_square {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x1: Array2<Q32> = args.array2();
        let x2: Array2<Q32> = args.array2();
        let gram = np::matmul(x1.view(), x2.t());
        let (s1, s2) = (row_square_sums(&x1), row_square_sums(&x2));
        Array2::from_shape_fn(gram.dim(), |(i, j)| gram[(i, j)] * -2 + s1[i] + s2[j])
    }
}
