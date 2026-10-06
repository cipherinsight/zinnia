//! Pythran wave_simulation: one step of a mass-spring string plucked in the
//! middle; returns the new position of the middle particle.
//!
//! ```python
//! def physics(masspoints, dt, plunk, which):
//!     ppos = masspoints[1]
//!     cpos = masspoints[0]
//!     N = cpos.shape[0]
//!     HOOKE_K = 2100000.0
//!     DAMPING = 0.0001
//!     MASS = 0.01
//!     force = np.zeros((N, 2))
//!     for i in range(1, N):
//!         dx, dy = cpos[i] - cpos[i - 1]
//!         dist = np.sqrt(dx ** 2 + dy ** 2)
//!         assert dist != 0
//!         fmag = -HOOKE_K * dist
//!         cosine = dx / dist
//!         sine = dy / dist
//!         fvec = np.array([fmag * cosine, fmag * sine])
//!         force[i - 1] -= fvec
//!         force[i] += fvec
//!     force[0] = force[-1] = (0, 0)
//!     force[which][1] += plunk
//!     accel = force / MASS
//!     npos = (2 - DAMPING) * cpos - (1 - DAMPING) * ppos + accel * dt ** 2
//!     masspoints[1] = cpos
//!     masspoints[0] = npos
//!
//! def test(PARTICLE_COUNT: Integer):
//!     SUBDIVISION = 300
//!     FRAMERATE = 60
//!     count = PARTICLE_COUNT
//!     width, height = (600, 200)
//!     masspoints = np.empty((2, count, 2), np.float64)
//!     initpos = np.zeros(count, np.float64)
//!     for i in range(1, count):
//!         initpos[i] = initpos[i - 1] + float(width) / count
//!     masspoints[:, :, 0] = initpos
//!     masspoints[:, :, 1] = height / 2
//!     f = 15
//!     plunk_pos = count // 2
//!     physics(masspoints, 1.0 / (SUBDIVISION * FRAMERATE), f, plunk_pos)
//!     return masspoints[0, count // 2]
//! ```
use crate::prelude::*;

/// `HOOKE_K = 2100000.0`.
const HOOKE_K: i64 = 2_100_000;
/// `DAMPING = 0.0001`: `round(0.0001 * 2^32)`.
const DAMPING: Q32 = Q32::from_bits(429_497);
/// `MASS = 0.01`: `round(0.01 * 2^32)`.
const MASS: Q32 = Q32::from_bits(42_949_673);
const SUBDIVISION: i64 = 300;
const FRAMERATE: i64 = 60;
const WIDTH: i64 = 600;
const HEIGHT: i64 = 200;
/// The pluck `f = 15`.
const PLUNK: i64 = 15;

/// `physics` for the one particle `test` returns: the spring forces on every
/// particle (with the kernel's `dist != 0` checks), then the update of
/// particle `which` only, since `test` reads no other row of `npos`.
fn physics_at(
    cpos: &Array2<Q32>,
    ppos: &Array2<Q32>,
    dt: Q32,
    plunk: i64,
    which: usize,
) -> Array1<Q32> {
    let n = cpos.nrows();
    let mut force = Array2::<Q32>::zeros((n, 2));
    for i in 1..n {
        let (dx, dy) = (
            cpos[(i, 0)] - cpos[(i - 1, 0)],
            cpos[(i, 1)] - cpos[(i - 1, 1)],
        );
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(dist != Q32::ZERO, "assert dist != 0");
        // `-HOOKE_K * dist`: a product with an integer-valued float, exact.
        let fmag = dist * -HOOKE_K;
        let fvec = [fmag * (dx / dist), fmag * (dy / dist)];
        for axis in 0..2 {
            force[(i - 1, axis)] -= fvec[axis];
            force[(i, axis)] += fvec[axis];
        }
    }
    // `force[0] = force[-1] = (0, 0)`, then the pluck.
    force.row_mut(0).fill(Q32::ZERO);
    force.row_mut(n - 1).fill(Q32::ZERO);
    force[(which, 1)] += Q32::from_int(plunk);
    let dt2 = dt * dt;
    Array1::from_shape_fn(2, |axis| {
        let accel = force[(which, axis)] / MASS;
        (Q32::from_int(2) - DAMPING) * cpos[(which, axis)]
            - (Q32::ONE - DAMPING) * ppos[(which, axis)]
            + accel * dt2
    })
}

pub mod test {
    use super::*;

    pub type Output = Array1<Q32>;

    #[allow(non_snake_case)]
    pub fn main(args: &mut Args<impl Input>) -> Output {
        let PARTICLE_COUNT: i64 = args.scalar();
        let count = usize::try_from(PARTICLE_COUNT).expect("negative dimensions are not allowed");
        // Both time levels start at the same positions: x evenly spaced
        // (accumulated as in the kernel), y = height / 2.
        let step = Q32::from_int(WIDTH) / PARTICLE_COUNT;
        let mut initpos = Array1::<Q32>::zeros(count);
        for i in 1..count {
            initpos[i] = initpos[i - 1] + step;
        }
        let half_height = Q32::from_int(HEIGHT) / 2;
        let positions =
            Array2::from_shape_fn(
                (count, 2),
                |(i, axis)| if axis == 0 { initpos[i] } else { half_height },
            );
        let dt = Q32::ONE / (SUBDIVISION * FRAMERATE);
        let middle = np_index(floor_div(PARTICLE_COUNT, 2), count);
        physics_at(&positions, &positions, dt, PLUNK, middle)
    }
}
