//! NPBench pythran/arc_distance: great-circle (haversine) distances between
//! the points `(theta_1, phi_1)` and `(theta_2, phi_2)`, elementwise.
//!
//! ```python
//! def arc_distance(theta_1: Float[Array, '100000'], phi_1: Float[Array, '100000'], theta_2: Float[Array, '100000'], phi_2: Float[Array, '100000']):
//!     temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
//!     distance_matrix = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
//!     return distance_matrix
//! ```
use crate::prelude::*;

/// `np.arctan2(y, x)`: the angle of `(x, y)` in `[-pi, pi]`, from `atan` of
/// the smaller over the larger magnitude (no overflow, and the argument stays
/// in `[0, 1]`). `arctan2(0, 0)` is 0.
fn atan2(y: Q32, x: Q32) -> Q32 {
    if x == Q32::ZERO && y == Q32::ZERO {
        return Q32::ZERO;
    }
    let (ay, ax) = (y.abs(), x.abs());
    let first_quadrant = if ay <= ax {
        (ay / ax).atan()
    } else {
        Q32::FRAC_PI_2 - (ax / ay).atan()
    };
    let angle = if x.is_negative() {
        Q32::PI - first_quadrant
    } else {
        first_quadrant
    };
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
        Zip::from(&theta_1)
            .and(&phi_1)
            .and(&theta_2)
            .and(&phi_2)
            .map_collect(|&t1, &p1, &t2, &p2| {
                let (dt, dp) = (((t2 - t1) / 2).sin(), ((p2 - p1) / 2).sin());
                let temp = dt * dt + t1.cos() * t2.cos() * (dp * dp);
                atan2(temp.sqrt(), (Q32::ONE - temp).sqrt()) * 2
            })
    }
}
