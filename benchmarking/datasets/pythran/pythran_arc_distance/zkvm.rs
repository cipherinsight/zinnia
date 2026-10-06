//! Pythran arc_distance: the great-circle angle between the points
//! `(theta_1[i], phi_1[i])` and `(theta_2[i], phi_2[i])` (haversine formula),
//! element-wise.
//!
//! ```python
//! def arc_distance(theta_1: Float[Array, '5'], phi_1: Float[Array, '5'], theta_2: Float[Array, '5'], phi_2: Float[Array, '5']):
//!     temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
//!     distance_matrix = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
//!     return distance_matrix
//! ```
use crate::prelude::*;

pub mod arc_distance {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let theta_1: Array1<Q32> = args.array1();
        let phi_1: Array1<Q32> = args.array1();
        let theta_2: Array1<Q32> = args.array1();
        let phi_2: Array1<Q32> = args.array1();
        let n = theta_1.len();
        assert!(
            [phi_1.len(), theta_2.len(), phi_2.len()].iter().all(|&len| len == n),
            "operands could not be broadcast together"
        );
        Array1::from_shape_fn(n, |i| {
            let dtheta = ((theta_2[i] - theta_1[i]) / 2).sin();
            let dphi = ((phi_2[i] - phi_1[i]) / 2).sin();
            let temp = dtheta * dtheta + theta_1[i].cos() * theta_2[i].cos() * (dphi * dphi);
            temp.sqrt().atan2((Q32::ONE - temp).sqrt()) * 2
        })
    }
}
