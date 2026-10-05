//! Pythran pythagorean_triples: the three children of every primitive
//! Pythagorean triple in Berggren's tree.
//!
//! ```python
//! def next_pythagorean_triples(previous: Integer[Array, '1 3']):
//!     matrices = np.array([[-1, 2, 2], [-2, 1, 2], [-2, 2, 3], [1, 2, 2], [2, 1, 2], [2, 2, 3], [1, -2, 2], [2, -1, 2], [2, -2, 3]])
//!     next_triples = np.transpose(np.dot(matrices, np.transpose(previous)))
//!     next_triples = next_triples.reshape((3 * previous.shape[0], previous.shape[1]))
//!     return next_triples
//! ```
//!
//! Row `p` of `np.transpose(np.dot(matrices, previous.T))` is
//! `matrices @ previous[p]`; the reshape splits it into three rows of three.
use crate::prelude::*;

/// Berggren's three matrices, stacked.
const MATRICES: [[i64; 3]; 9] = [
    [-1, 2, 2],
    [-2, 1, 2],
    [-2, 2, 3],
    [1, 2, 2],
    [2, 1, 2],
    [2, 2, 3],
    [1, -2, 2],
    [2, -1, 2],
    [2, -2, 3],
];

pub mod next_pythagorean_triples {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let previous: Array2<i64> = args.array2();
        // The product needs 3 columns, and the reshape to (3 * rows, columns)
        // of the rows x 9 product then holds.
        assert_eq!(previous.ncols(), 3, "shapes not aligned");
        let rows = previous.nrows();
        Array2::from_shape_fn((3 * rows, 3), |(r, c)| {
            let (triple, matrix_row) = (previous.row(r / 3), &MATRICES[3 * (r % 3) + c]);
            matrix_row.iter().zip(triple).map(|(&m, &t)| m * t).sum()
        })
    }
}
