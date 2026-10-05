//! Pythran diffusion: forward-Euler steps of the 2-D heat equation, in place
//! on `u`, with `tempU` as the scratch grid.
//!
//! ```python
//! def diffuseNumpy(u: Float[Array, '128 128'], tempU: Float[Array, '128 128'], iterNum: Integer):
//!     mu = 0.1
//!     for n in range(iterNum):
//!         tempU[1:-1, 1:-1] = u[1:-1, 1:-1] + mu * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1] + u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
//!         u[:, :] = tempU[:, :]
//!         tempU[:, :] = 0.0
//! ```
use crate::prelude::*;

/// `round(0.1 * 2^32)`.
const MU: Q32 = Q32::from_bits(429_496_730);

#[allow(non_snake_case)]
pub mod diffuseNumpy {
    use super::*;

    // outputs: mutated u, tempU
    pub type Output = (Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut u: Array2<Q32> = args.array2();
        let mut temp_u: Array2<Q32> = args.array2();
        let iter_num: i64 = args.scalar();
        let (rows, cols) = u.dim();
        for _ in 0..iter_num {
            // The slices 1:-1, 2: and 0:-2 of each axis.
            Zip::from(temp_u.slice_mut(s![1..rows - 1, 1..cols - 1]))
                .and(u.slice(s![1..rows - 1, 1..cols - 1]))
                .and(u.slice(s![2..rows, 1..cols - 1]))
                .and(u.slice(s![0..rows - 2, 1..cols - 1]))
                .and(u.slice(s![1..rows - 1, 2..cols]))
                .and(u.slice(s![1..rows - 1, 0..cols - 2]))
                .for_each(|t, &centre, &south, &north, &east, &west| {
                    *t = centre + MU * (south - centre * 2 + north + east - centre * 2 + west);
                });
            // The first copy also brings tempU's input border into u.
            u.assign(&temp_u);
            temp_u.fill(Q32::ZERO);
        }
        (u, temp_u)
    }
}
