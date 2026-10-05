//! Pythran calculate_u: explicit time stepping of the 2-D wave equation with
//! a variable coefficient `k`, until `t` passes `t_stop`.
//!
//! ```python
//! @requires(lambda t, t_stop, dt, dx, dy, u, um, k: dt != 0.0)
//! @requires(lambda t, t_stop, dt, dx, dy, u, um, k: dx != 0.0)
//! @requires(lambda t, t_stop, dt, dx, dy, u, um, k: dy != 0.0)
//! def timeloop(t: Float, t_stop: Float, dt: Float, dx: Float, dy: Float, u: Integer[Array, '5 10'], um: Integer[Array, '5 10'], k: Integer[Array, '5 10']):
//!     while t <= t_stop:
//!         t += dt
//!         new_u = calculate_u(dt, dx, dy, u, um, k)
//!         um = u
//!         u = new_u
//!     return u
//!
//! def calculate_u(dt, dx, dy, u, um, k):
//!     up = [[0.0] * len(u[0]) for i in range(len(u))]
//!     for i in range(1, len(u) - 1):
//!         for j in range(1, len(u[0]) - 1):
//!             up[i][j] = 2 * u[i][j] - um[i][j] + (dt / dx) ** 2 * (0.5 * (k[i + 1][j] + k[i][j]) * (u[i + 1][j] - u[i][j]) - 0.5 * (k[i][j] + k[i - 1][j]) * (u[i][j] - u[i - 1][j])) + (dt / dy) ** 2 * (0.5 * (k[i][j + 1] + k[i][j]) * (u[i][j + 1] - u[i][j]) - 0.5 * (k[i][j] + k[i][j - 1]) * (u[i][j] - u[i][j - 1]))
//!     return up
//! ```
//!
//! The returned `u` is the integer input when the loop never runs, and
//! otherwise the last `calculate_u` result, a list of float rows (the
//! boundary is 0.0).
use crate::prelude::*;

/// `0.5`, exact in Q32.
const HALF: Q32 = Q32::from_bits(1 << 31);

/// One `calculate_u` step: the new interior from `u` and `um`, with a zero
/// boundary. `cx`, `cy` are `(dt / dx) ** 2` and `(dt / dy) ** 2`. The
/// integer inputs enter as exact Q32 values (`2 * u - um` and the integer
/// differences are exact either way).
fn calculate_u(
    cx: Q32,
    cy: Q32,
    u: &Array2<Q32>,
    um: &Array2<Q32>,
    k: &Array2<i64>,
) -> Array2<Q32> {
    let (rows, cols) = u.dim();
    let mut up = Array2::<Q32>::zeros((rows, cols));
    // `0.5 * (k1 + k2) * (u1 - u2)`: the integer sum of `k`, halved exactly.
    let flux = |k1: i64, k2: i64, du: Q32| HALF * Q32::from_int(k1 + k2) * du;
    for i in 1..rows.saturating_sub(1) {
        for j in 1..cols.saturating_sub(1) {
            let (kc, uc) = (k[(i, j)], u[(i, j)]);
            let along_x = flux(k[(i + 1, j)], kc, u[(i + 1, j)] - uc)
                - flux(kc, k[(i - 1, j)], uc - u[(i - 1, j)]);
            let along_y = flux(k[(i, j + 1)], kc, u[(i, j + 1)] - uc)
                - flux(kc, k[(i, j - 1)], uc - u[(i, j - 1)]);
            up[(i, j)] = uc * 2 - um[(i, j)] + cx * along_x + cy * along_y;
        }
    }
    up
}

pub mod timeloop {
    use super::*;

    /// The integer `u` if the loop never runs, else the float result.
    pub type Output = (Option<Array2<i64>>, Option<Array2<Q32>>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let [mut t, t_stop, dt, dx, dy]: [Q32; 5] = core::array::from_fn(|_| args.scalar());
        let u: Array2<i64> = args.array2();
        let um: Array2<i64> = args.array2();
        let k: Array2<i64> = args.array2();
        assert!(dt != Q32::ZERO, "requires dt != 0.0");
        assert!(dx != Q32::ZERO, "requires dx != 0.0");
        assert!(dy != Q32::ZERO, "requires dy != 0.0");
        if t > t_stop {
            return (Some(u), None);
        }
        assert!(u.nrows() > 0, "u[0] of an empty array");
        let (ratio_x, ratio_y) = (dt / dx, dt / dy);
        let (cx, cy) = (ratio_x * ratio_x, ratio_y * ratio_y);
        let mut u = u.mapv(Q32::from_int);
        let mut um = um.mapv(Q32::from_int);
        while t <= t_stop {
            t += dt;
            let new_u = calculate_u(cx, cy, &u, &um, &k);
            um = core::mem::replace(&mut u, new_u);
        }
        (None, Some(u))
    }
}
