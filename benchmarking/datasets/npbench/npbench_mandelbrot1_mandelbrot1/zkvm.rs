//! NPBench mandelbrot1: the Mandelbrot iteration on a `yn x xn` grid; a point
//! is updated while `|z| < horizon` and records the last iteration at which
//! it was updated.
//!
//! ```python
//! def mandelbrot(xmin: Float, xmax: Float, ymin: Float, ymax: Float, xn: Integer, yn: Integer, maxiter: Integer, horizon: Float=2.0):
//!     X = np.linspace(xmin, xmax, xn, dtype=np.float64)
//!     Y = np.linspace(ymin, ymax, yn, dtype=np.float64)
//!     C = X + Y[:, None] * 1j
//!     N = np.zeros(C.shape, dtype=np.int64)
//!     Z = np.zeros(C.shape, dtype=np.complex128)
//!     for n in range(maxiter):
//!         I = np.less(abs(Z), horizon)
//!         N[I] = n
//!         Z[I] = Z[I] ** 2 + C[I]
//!     N[N == maxiter - 1] = 0
//!     return (Z, N)
//! ```
//!
//! The points evolve independently, so each is iterated on its own; once
//! `|z| >= horizon` it is never updated again, so its loop stops there.
use crate::prelude::*;

pub mod mandelbrot {
    use super::*;

    /// `(Z, N)`, both `yn x xn`.
    pub type Output = (Array2<C32>, Array2<i64>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let [xmin, xmax, ymin, ymax]: [Q32; 4] = core::array::from_fn(|_| args.scalar());
        let xn: i64 = args.scalar();
        let yn: i64 = args.scalar();
        let maxiter: i64 = args.scalar();
        let horizon: Q32 = args.scalar();
        let xs = np::linspace(xmin, xmax, xn);
        let ys = np::linspace(ymin, ymax, yn);
        let mut z_out = Array2::<C32>::zeros((ys.len(), xs.len()));
        let mut n_out = Array2::<i64>::zeros((ys.len(), xs.len()));
        for (r, &y) in ys.iter().enumerate() {
            for (col, &x) in xs.iter().enumerate() {
                let c = C32::new(x, y);
                let (mut z, mut n) = (C32::ZERO, 0);
                for it in 0..maxiter {
                    if z.abs() >= horizon {
                        break;
                    }
                    n = it;
                    z = z * z + c; // `Z ** 2` is `np.square`
                }
                // `N[N == maxiter - 1] = 0`
                n_out[(r, col)] = if n == maxiter - 1 { 0 } else { n };
                z_out[(r, col)] = z;
            }
        }
        (z_out, n_out)
    }
}
