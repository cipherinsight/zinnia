//! NPBench polybench/floyd_warshall: all-pairs shortest paths, in place.
//!
//! ```python
//! def kernel(path: Integer[Array, '200 200']):
//!     for k in range(path.shape[0]):
//!         path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated path
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut path: Array2<i64> = args.array2();
        for k in 0..path.nrows() {
            // `np.add.outer` reads column k and row k before the update
            // (they change within the step if path[k, k] < 0).
            let (column, row) = (path.column(k).to_owned(), path.row(k).to_owned());
            for ((i, j), p) in path.indexed_iter_mut() {
                *p = (*p).min(column[i].wrapping_add(row[j]));
            }
        }
        path
    }
}
