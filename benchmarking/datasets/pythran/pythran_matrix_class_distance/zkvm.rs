//! Pythran matrix_class_distance: for each class `i < iterations`, the summed
//! Euclidean distance of the rows of `dat` in class `i` to the class point
//! `dat_points[i]`, added over the classes.
//!
//! ```python
//! def matrix_class_distance(dat: Float[Array, '200 10'], dat_filter: Integer[Array, '200'], dat_points: Float[Array, '200 10'], iterations: Integer):
//!     aggregation = 0
//!     for i in range(iterations):
//!         aggregation += np.sum(np.linalg.norm(dat[dat_filter == i] - dat_points[i], axis=1))
//!     return aggregation
//! ```
use crate::prelude::*;

pub mod matrix_class_distance {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let dat: Array2<Q32> = args.array2();
        let dat_filter: Array1<i64> = args.array1();
        let dat_points: Array2<Q32> = args.array2();
        let iterations: i64 = args.scalar();
        assert_eq!(dat_filter.len(), dat.nrows(), "boolean index does not match the rows of dat");
        let mut aggregation = Q32::ZERO;
        for i in 0..iterations.max(0) {
            let point = dat_points.row(np_index(i, dat_points.nrows()));
            assert_eq!(point.len(), dat.ncols(), "operands could not be broadcast together");
            for (row, &class) in dat.rows().into_iter().zip(&dat_filter) {
                if class != i {
                    continue;
                }
                // `np.linalg.norm(..., axis=1)`: square each difference, sum, root.
                let squares: Q32 = row.iter().zip(point).map(|(&x, &p)| (x - p) * (x - p)).sum();
                aggregation += squares.sqrt();
            }
        }
        aggregation
    }
}
