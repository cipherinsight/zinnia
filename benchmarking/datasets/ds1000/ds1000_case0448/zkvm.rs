//! DS-1000 case 0448: pair up the entries of two matrices,
//! `dists[i][j] = [x_dists[i][j], y_dists[i][j]]`.
//!
//! ```python
//! def kernel(x_dists: Float[Array, "3 4"], y_dists: Float[Array, "3 4"]):
//!     dists = np.vstack(([x_dists.T], [y_dists.T])).T
//!     return dists
//! ```
use crate::prelude::*;

const N: usize = 3 * 4;

pub type Output = Vec<Q32>;

/// The `vstack` has shape `(2, 4, 3)` with `[k][j][i]` the entry `(i, j)` of
/// the `k`-th matrix; the final `.T` reverses the axes to `(3, 4, 2)`.
pub fn main(input: &mut impl Input) -> Output {
    let x_dists: Vec<Q32> = input.read_vec(N);
    let y_dists: Vec<Q32> = input.read_vec(N);
    x_dists.iter().zip(&y_dists).flat_map(|(&x, &y)| [x, y]).collect()
}
