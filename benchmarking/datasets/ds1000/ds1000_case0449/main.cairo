//! DS-1000 case 0449: pair up the entries of two matrices,
//! `dists[i][j] = [x_dists[i][j], y_dists[i][j]]`.
//!
//!     def kernel(x_dists: Float[Array, "3 4"], y_dists: Float[Array, "3 4"]):
//!         dists = np.vstack(([x_dists.T], [y_dists.T])).T
//!         return dists
// output: [q32]
use crate::q32::Q32;

const N: u32 = 3 * 4;

/// The `vstack` has shape `(2, 4, 3)` with `[k][j][i]` the entry `(i, j)` of
/// the `k`-th matrix; the final `.T` reverses the axes to `(3, 4, 2)`.
#[executable]
fn main(x_dists: Array<Q32>, y_dists: Array<Q32>) -> Array<Q32> {
    assert!(x_dists.len() == N && y_dists.len() == N, "inputs have the wrong length");
    let mut dists = array![];
    for k in 0..N {
        dists.append(*x_dists.at(k));
        dists.append(*y_dists.at(k));
    }
    dists
}
