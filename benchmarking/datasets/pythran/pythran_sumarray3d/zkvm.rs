//! Pythran sumarray3d: for every point, the weighted sum of inverse distances
//! to a set of positions and the weighted sum of the separation vectors.
//!
//! ```python
//! def summation(pos: Float[Array, '30 3'], weights: Float[Array, '30'], points: Float[Array, '30 3']):
//!     n_points = len(points)
//!     n_weights = len(weights)
//!     sum_array3d = np.zeros((n_points, 3))
//!
//!     def compute(i):
//!         pxi = points[i, 0]
//!         pyi = points[i, 1]
//!         pzi = points[i, 2]
//!         total = 0.0
//!         for j in range(n_weights):
//!             weight_j = weights[j]
//!             xj = pos[j, 0]
//!             yj = pos[j, 1]
//!             zj = pos[j, 2]
//!             dx = pxi - pos[j, 0]
//!             dy = pyi - pos[j, 1]
//!             dz = pzi - pos[j, 2]
//!             dr = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
//!             total += weight_j * dr
//!             sum_array3d[i, 0] += weight_j * dx
//!             sum_array3d[i, 1] += weight_j * dy
//!             sum_array3d[i, 2] += weight_j * dz
//!         return total
//!     sum_array = np.array([compute(i) for i in range(n_points)])
//!     return (sum_array, sum_array3d)
//! ```
use crate::prelude::*;

pub mod summation {
    use super::*;

    pub type Output = (Array1<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let pos: Array2<Q32> = args.array2();
        let weights: Array1<Q32> = args.array1();
        let points: Array2<Q32> = args.array2();
        let n_points = points.nrows();
        let mut sum_array = Array1::<Q32>::zeros(n_points);
        let mut sum_array3d = Array2::<Q32>::zeros((n_points, 3));
        for i in 0..n_points {
            let (pxi, pyi, pzi) = (points[(i, 0)], points[(i, 1)], points[(i, 2)]);
            let mut total = Q32::ZERO;
            for (j, &weight_j) in weights.iter().enumerate() {
                let (dx, dy, dz) = (pxi - pos[(j, 0)], pyi - pos[(j, 1)], pzi - pos[(j, 2)]);
                let dr = Q32::ONE / (dx * dx + dy * dy + dz * dz).sqrt();
                total += weight_j * dr;
                sum_array3d[(i, 0)] += weight_j * dx;
                sum_array3d[(i, 1)] += weight_j * dy;
                sum_array3d[(i, 2)] += weight_j * dz;
            }
            sum_array[i] = total;
        }
        (sum_array, sum_array3d)
    }
}
