//! Pythran periodic_dist: all pairwise separations and distances of particles
//! in a box of side `L`, with optional periodic wrapping per axis.
//!
//! ```python
//! def dist(x: Float[Array, '200'], y: Float[Array, '200'], z: Float[Array, '200'], L: Integer, periodicX: Boolean, periodicY: Boolean, periodicZ: Boolean):
//!     N = len(x)
//!     xtemp = np.tile(x, (N, 1))
//!     dx = xtemp - xtemp.T
//!     ytemp = np.tile(y, (N, 1))
//!     dy = ytemp - ytemp.T
//!     ztemp = np.tile(z, (N, 1))
//!     dz = ztemp - ztemp.T
//!     if periodicX:
//!         dx[dx > L / 2] = dx[dx > L / 2] - L
//!         dx[dx < -L / 2] = dx[dx < -L / 2] + L
//!     if periodicY:
//!         dy[dy > L / 2] = dy[dy > L / 2] - L
//!         dy[dy < -L / 2] = dy[dy < -L / 2] + L
//!     if periodicZ:
//!         dz[dz > L / 2] = dz[dz > L / 2] - L
//!         dz[dz < -L / 2] = dz[dz < -L / 2] + L
//!     d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
//!     d[d == 0] = -1
//!     return (d, dx, dy, dz)
//! ```
use crate::prelude::*;

/// `np.tile(v, (n, 1))` minus its transpose, `(i, j) -> v[j] - v[i]`, then,
/// if `periodic`, the two wrapping passes: entries above `L / 2` lose `L`,
/// and then entries below `-L / 2` gain `L`.
fn separations(v: &Array1<Q32>, n: usize, l: i64, periodic: bool) -> Array2<Q32> {
    assert_eq!(v.len(), n, "operands could not be broadcast together");
    let side = Q32::from_int(l);
    let (upper, lower) = (side / 2, Q32::from_int(-l) / 2);
    Array2::from_shape_fn((n, n), |(i, j)| {
        let mut s = v[j] - v[i];
        if periodic {
            if s > upper {
                s -= side;
            }
            if s < lower {
                s += side;
            }
        }
        s
    })
}

pub mod dist {
    use super::*;

    pub type Output = (Array2<Q32>, Array2<Q32>, Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        let y: Array1<Q32> = args.array1();
        let z: Array1<Q32> = args.array1();
        let l: i64 = args.scalar();
        let periodic_x: bool = args.scalar();
        let periodic_y: bool = args.scalar();
        let periodic_z: bool = args.scalar();
        let n = x.len();
        let dx = separations(&x, n, l, periodic_x);
        let dy = separations(&y, n, l, periodic_y);
        let dz = separations(&z, n, l, periodic_z);
        let d = Zip::from(&dx).and(&dy).and(&dz).map_collect(|&a, &b, &c| {
            let r = (a * a + b * b + c * c).sqrt();
            if r == Q32::ZERO {
                -Q32::ONE
            } else {
                r
            }
        });
        (d, dx, dy, dz)
    }
}
