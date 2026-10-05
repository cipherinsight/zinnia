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

/// `np.arctan2(y, x)`: the angle of `(x, y)` in `[-pi, pi]`, from
/// [`Q32::atan`] of the smaller over the larger magnitude (`arctan2(0, 0)` is
/// 0).
fn atan2(y: Q32, x: Q32) -> Q32 {
    let (ax, ay) = (x.abs(), y.abs());
    if ax == Q32::ZERO && ay == Q32::ZERO {
        return Q32::ZERO;
    }
    let first_quadrant =
        if ay <= ax { (ay / ax).atan() } else { Q32::FRAC_PI_2 - (ax / ay).atan() };
    let angle = if x.is_negative() { Q32::PI - first_quadrant } else { first_quadrant };
    if y.is_negative() {
        -angle
    } else {
        angle
    }
}

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
            atan2(temp.sqrt(), (Q32::ONE - temp).sqrt()) * 2
        })
    }
}
