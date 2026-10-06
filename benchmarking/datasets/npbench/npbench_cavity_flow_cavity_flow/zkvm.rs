//! NPBench cavity_flow: lid-driven cavity flow, finite differences for the
//! Navier-Stokes equations with a pressure Poisson solve per time step, in
//! place. `p` is updated in place too, by the helper `pressure_poisson`; the
//! body itself writes `u` and `v`.
//!
//! ```python
//! def build_up_b(b, rho, dt, u, v, dx, dy):
//!     b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 - 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2)
//!
//! def pressure_poisson(nit, p, dx, dy, b):
//!     pn = np.empty_like(p)
//!     pn = p.copy()
//!     for q in range(nit):
//!         pn = p.copy()
//!         p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) - dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
//!         p[:, -1] = p[:, -2]
//!         p[0, :] = p[1, :]
//!         p[:, 0] = p[:, 1]
//!         p[-1, :] = 0
//!
//! @requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: 0 <= nt <= 1024)
//! @requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: 0 <= nit <= 1024)
//! @requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: dt != 0.0)
//! @requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: dx != 0.0)
//! @requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: dy != 0.0)
//! def cavity_flow(nx: Integer, ny: Integer, nt: Integer, nit: Integer, u: Float[Array, '61 61'], v: Float[Array, '61 61'], dt: Float, dx: Float, dy: Float, p: Float[Array, '61 61'], rho: Float, nu: Float):
//!     un = np.empty_like(u)
//!     vn = np.empty_like(v)
//!     b = np.zeros((ny, nx))
//!     for n in range(nt):
//!         un = u.copy()
//!         vn = v.copy()
//!         build_up_b(b, rho, dt, u, v, dx, dy)
//!         pressure_poisson(nit, p, dx, dy, b)
//!         u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
//!         v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
//!         u[0, :] = 0
//!         u[:, 0] = 0
//!         u[:, -1] = 0
//!         u[-1, :] = 1
//!         v[0, :] = 0
//!         v[-1, :] = 0
//!         v[:, 0] = 0
//!         v[:, -1] = 0
//! ```
//!
//! For an interior point `c = (i, j)`, the neighbours `e`/`w` are
//! `(i, j + 1)`/`(i, j - 1)` (slices `[1:-1, 2:]`/`[1:-1, 0:-2]`) and `s`/`n`
//! are `(i + 1, j)`/`(i - 1, j)` (`[2:, 1:-1]`/`[0:-2, 1:-1]`).
use crate::prelude::*;

/// The interior points `(i, j)` of an `h x w` grid (`[1:-1, 1:-1]`).
fn interior((h, w): (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    (1..h.saturating_sub(1)).flat_map(move |i| (1..w.saturating_sub(1)).map(move |j| (i, j)))
}

/// `build_up_b`: the source term of the pressure equation, on the interior.
fn build_up_b(b: &mut Array2<Q32>, rho: Q32, dt: Q32, u: &Array2<Q32>, v: &Array2<Q32>, dx: Q32, dy: Q32) {
    let inv_dt = Q32::ONE / dt;
    let (two_dx, two_dy) = (dx * 2, dy * 2);
    for (i, j) in interior(u.dim()) {
        let du_dx = (u[(i, j + 1)] - u[(i, j - 1)]) / two_dx;
        let dv_dy = (v[(i + 1, j)] - v[(i - 1, j)]) / two_dy;
        let du_dy_dv_dx = (u[(i + 1, j)] - u[(i - 1, j)]) / two_dy * (v[(i, j + 1)] - v[(i, j - 1)]) / two_dx;
        b[(i, j)] = rho * (inv_dt * (du_dx + dv_dy) - du_dx.powi(2) - du_dy_dv_dx * 2 - dv_dy.powi(2));
    }
}

/// `pressure_poisson`: `nit` Jacobi sweeps of the pressure equation, with
/// the kernel's boundary conditions after each.
fn pressure_poisson(nit: i64, p: &mut Array2<Q32>, dx: Q32, dy: Q32, b: &Array2<Q32>) {
    let (dx2, dy2) = (dx.powi(2), dy.powi(2));
    let denominator = (dx2 + dy2) * 2;
    let source_weight = dx2 * dy2 / denominator;
    let (h, w) = p.dim();
    for _ in 0..nit {
        let pn = p.clone();
        for (i, j) in interior((h, w)) {
            p[(i, j)] = ((pn[(i, j + 1)] + pn[(i, j - 1)]) * dy2 + (pn[(i + 1, j)] + pn[(i - 1, j)]) * dx2)
                / denominator
                - source_weight * b[(i, j)];
        }
        for i in 0..h {
            p[(i, w - 1)] = p[(i, w - 2)];
        }
        for j in 0..w {
            p[(0, j)] = p[(1, j)];
        }
        for i in 0..h {
            p[(i, 0)] = p[(i, 1)];
        }
        p.row_mut(h - 1).fill(Q32::ZERO);
    }
}

pub mod cavity_flow {
    use super::*;

    // outputs: mutated u, v
    pub type Output = (Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let nx: i64 = args.scalar();
        let ny: i64 = args.scalar();
        let nt: i64 = args.scalar();
        let nit: i64 = args.scalar();
        let mut u: Array2<Q32> = args.array2();
        let mut v: Array2<Q32> = args.array2();
        let dt: Q32 = args.scalar();
        let dx: Q32 = args.scalar();
        let dy: Q32 = args.scalar();
        let mut p: Array2<Q32> = args.array2();
        let rho: Q32 = args.scalar();
        let nu: Q32 = args.scalar();
        assert!((0..=1024).contains(&nt), "requires 0 <= nt <= 1024");
        assert!((0..=1024).contains(&nit), "requires 0 <= nit <= 1024");
        assert!(dt != Q32::ZERO, "requires dt != 0.0");
        assert!(dx != Q32::ZERO, "requires dx != 0.0");
        assert!(dy != Q32::ZERO, "requires dy != 0.0");
        let dim = |d: i64| usize::try_from(d).expect("negative dimensions are not allowed");
        let mut b = Array2::zeros((dim(ny), dim(nx)));
        let shape = u.dim();
        assert!(v.dim() == shape && p.dim() == shape && b.dim() == shape, "operands could not be broadcast together");
        let (rows, cols) = shape;

        let (dx2, dy2) = (dx.powi(2), dy.powi(2));
        let (dt_dx2, dt_dy2) = (dt / dx2, dt / dy2);
        let pressure_x = dt / (rho * 2 * dx);
        let pressure_y = dt / (rho * 2 * dy);
        for _ in 0..nt {
            let un = u.clone();
            let vn = v.clone();
            build_up_b(&mut b, rho, dt, &u, &v, dx, dy);
            pressure_poisson(nit, &mut p, dx, dy, &b);
            for (i, j) in interior(shape) {
                let (c, e, w, s, n) = ((i, j), (i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j));
                u[c] = un[c] - un[c] * dt / dx * (un[c] - un[w]) - vn[c] * dt / dy * (un[c] - un[n])
                    - pressure_x * (p[e] - p[w])
                    + nu * (dt_dx2 * (un[e] - un[c] * 2 + un[w]) + dt_dy2 * (un[s] - un[c] * 2 + un[n]));
                v[c] = vn[c] - un[c] * dt / dx * (vn[c] - vn[w]) - vn[c] * dt / dy * (vn[c] - vn[n])
                    - pressure_y * (p[s] - p[n])
                    + nu * (dt_dx2 * (vn[e] - vn[c] * 2 + vn[w]) + dt_dy2 * (vn[s] - vn[c] * 2 + vn[n]));
            }
            u.row_mut(0).fill(Q32::ZERO);
            u.column_mut(0).fill(Q32::ZERO);
            u.column_mut(cols - 1).fill(Q32::ZERO);
            u.row_mut(rows - 1).fill(Q32::ONE);
            v.row_mut(0).fill(Q32::ZERO);
            v.row_mut(rows - 1).fill(Q32::ZERO);
            v.column_mut(0).fill(Q32::ZERO);
            v.column_mut(cols - 1).fill(Q32::ZERO);
        }
        (u, v)
    }
}
