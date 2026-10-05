//! Pythran laplace: `Niter` in-place (Gauss-Seidel) Jacobi sweeps of the
//! Laplace equation on an `N x N` grid whose first row is held at 1.
//!
//! ```python
//! def update(u):
//!     dx = 0.1
//!     dy = 0.1
//!     dx2 = dx * dx
//!     dy2 = dy * dy
//!     nx, ny = (len(u), len(u[0]))
//!     for i in range(1, nx - 1):
//!         for j in range(1, ny - 1):
//!             u[i][j] = ((u[i + 1][j] + u[i - 1][j]) * dy2 + (u[i][j + 1] + u[i][j - 1]) * dx2) / (2 * (dx2 + dy2))
//!
//! def calc(N: Integer, Niter: Integer=100):
//!     u = [[0] * N for _ in range(N)]
//!     u[0] = [1] * N
//!     for i in range(Niter):
//!         update(u)
//!     return u
//! ```
use crate::prelude::*;

/// `0.1`, quantized.
const DX: Q32 = Q32::from_bits(429_496_730);
const DY: Q32 = DX;

/// One sweep of `update`: each interior cell, in row-major order, becomes the
/// weighted mean of its four neighbours (already updated ones included).
fn update(u: &mut Array2<Q32>) {
    let (dx2, dy2) = (DX * DX, DY * DY);
    let denominator = (dx2 + dy2) * 2;
    let (nx, ny) = u.dim();
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            u[(i, j)] = ((u[(i + 1, j)] + u[(i - 1, j)]) * dy2
                + (u[(i, j + 1)] + u[(i, j - 1)]) * dx2)
                / denominator;
        }
    }
}

pub mod calc {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let niter: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= N <= 1024");
        assert!((0..=1024).contains(&niter), "requires 0 <= Niter <= 1024");
        let mut u = Array2::<Q32>::zeros((n as usize, n as usize));
        // `u[0] = [1] * N` raises IndexError for N = 0, as `row_mut(0)` panics.
        u.row_mut(0).fill(Q32::ONE);
        for _ in 0..niter {
            update(&mut u);
        }
        u
    }
}
