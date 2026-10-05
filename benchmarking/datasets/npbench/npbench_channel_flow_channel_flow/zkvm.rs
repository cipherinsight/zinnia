//! NPBench channel_flow: 2-D Navier-Stokes channel flow (periodic in x,
//! walls in y, driven by the force `F`), stepped until the relative change of
//! `sum(u)` is at most 0.001; returns the number of steps.
//!
//! ```python
//! def build_up_b(rho, dt, dx, dy, u, v):
//!     b = np.zeros_like(u)
//!     b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 - 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2)
//!     b[1:-1, -1] = (the same with column -1 between columns 0 and -2)
//!     b[1:-1, 0] = (the same with column 0 between columns 1 and -1)
//!     return b
//!
//! def pressure_poisson_periodic(nit, p, dx, dy, b):
//!     pn = np.empty_like(p)
//!     for q in range(nit):
//!         pn = p.copy()
//!         p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) - dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
//!         p[1:-1, -1] = (the same with column -1 between columns 0 and -2)
//!         p[1:-1, 0] = (the same with column 0 between columns 1 and -1)
//!         p[-1, :] = p[-2, :]
//!         p[0, :] = p[1, :]
//!
//! def channel_flow(nit: Integer, u: Float[Array, '61 61'], v: Float[Array, '61 61'], dt: Float, dx: Float, dy: Float, p: Float[Array, '61 61'], rho: Float, nu: Float, F: Float):
//!     udiff = 1
//!     stepcount = 0
//!     while udiff > 0.001:
//!         un = u.copy()
//!         vn = v.copy()
//!         b = build_up_b(rho, dt, dx, dy, u, v)
//!         pressure_poisson_periodic(nit, p, dx, dy, b)
//!         u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + F * dt
//!         v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
//!         u[1:-1, -1] = ..., u[1:-1, 0] = ..., v[1:-1, -1] = ..., v[1:-1, 0] = ...
//!                       (the same for the first and last columns, periodic)
//!         u[0, :] = 0
//!         u[-1, :] = 0
//!         v[0, :] = 0
//!         v[-1, :] = 0
//!         udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
//!         stepcount += 1
//!     return stepcount
//! ```
//!
//! The first- and last-column statements are the interior formula with the
//! column axis periodic, so every row update below runs over all columns with
//! `east_west`. Each statement reads only copies (`un`, `vn`, `pn`) or arrays
//! it does not write (`b`, `p`), so the element order does not matter.
use crate::prelude::*;

/// `0.001`, quantized.
const TOLERANCE: Q32 = Q32::from_bits(4_294_967);

/// The columns east and west of column `j` on the periodic column axis of
/// width `w`: `j + 1` and `j - 1`, with the last column's east neighbour
/// column 0 and the first column's west neighbour column -1.
fn east_west(j: usize, w: usize) -> (usize, usize) {
    ((j + 1) % w, (j + w - 1) % w)
}

/// The interior rows `1..h-1` (`1:-1`).
fn interior(h: usize) -> core::ops::Range<usize> {
    1..h.saturating_sub(1)
}

fn build_up_b(
    rho: Q32,
    dt: Q32,
    dx: Q32,
    dy: Q32,
    u: &Array2<Q32>,
    v: &Array2<Q32>,
) -> Array2<Q32> {
    let (h, w) = u.dim();
    let mut b = Array2::zeros((h, w));
    let inv_dt = Q32::ONE / dt;
    let (two_dx, two_dy) = (dx * 2, dy * 2);
    for i in interior(h) {
        for j in 0..w {
            let (east, west) = east_west(j, w);
            let du_dx = (u[(i, east)] - u[(i, west)]) / two_dx;
            let dv_dy = (v[(i + 1, j)] - v[(i - 1, j)]) / two_dy;
            let cross =
                (u[(i + 1, j)] - u[(i - 1, j)]) / two_dy * (v[(i, east)] - v[(i, west)]) / two_dx;
            b[(i, j)] =
                rho * (inv_dt * (du_dx + dv_dy) - du_dx * du_dx - cross * 2 - dv_dy * dv_dy);
        }
    }
    b
}

fn pressure_poisson_periodic(nit: i64, p: &mut Array2<Q32>, dx: Q32, dy: Q32, b: &Array2<Q32>) {
    let (h, w) = p.dim();
    let (dx2, dy2) = (dx * dx, dy * dy);
    let denominator = (dx2 + dy2) * 2;
    let source = dx2 * dy2 / denominator;
    for _ in 0..nit {
        let pn = p.clone();
        for i in interior(h) {
            for j in 0..w {
                let (east, west) = east_west(j, w);
                p[(i, j)] = ((pn[(i, east)] + pn[(i, west)]) * dy2
                    + (pn[(i + 1, j)] + pn[(i - 1, j)]) * dx2)
                    / denominator
                    - source * b[(i, j)];
            }
        }
        // p[-1, :] = p[-2, :]; p[0, :] = p[1, :]
        let (last, second_last) = (np_index(-1, h), np_index(-2, h));
        let row = p.row(second_last).to_owned();
        p.row_mut(last).assign(&row);
        let row = p.row(1).to_owned();
        p.row_mut(0).assign(&row);
    }
}

pub mod channel_flow {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let nit: i64 = args.scalar();
        let mut u: Array2<Q32> = args.array2();
        let mut v: Array2<Q32> = args.array2();
        let dt: Q32 = args.scalar();
        let dx: Q32 = args.scalar();
        let dy: Q32 = args.scalar();
        let mut p: Array2<Q32> = args.array2();
        let rho: Q32 = args.scalar();
        let nu: Q32 = args.scalar();
        let force: Q32 = args.scalar();
        assert!((0..=1024).contains(&nit), "requires 0 <= nit <= 1024");
        assert!(dt != Q32::ZERO, "requires dt != 0.0");
        assert!(dx != Q32::ZERO, "requires dx != 0.0");
        assert!(dy != Q32::ZERO, "requires dy != 0.0");
        assert!(rho != Q32::ZERO, "requires rho != 0.0");
        assert!(u.dim() == v.dim() && p.dim() == u.dim(), "u, v and p differ in shape");
        let (h, w) = u.dim();
        // The first step always runs, and its `b[1:-1, -1]` reads column -2.
        assert!(w >= 2, "index -2 is out of bounds for axis 1");

        let dt_2rho_dx = dt / (rho * 2 * dx);
        let dt_2rho_dy = dt / (rho * 2 * dy);
        let (dt_dx2, dt_dy2) = (dt / (dx * dx), dt / (dy * dy));
        let force_dt = force * dt;

        let mut udiff = Q32::ONE;
        let mut stepcount = 0;
        while udiff > TOLERANCE {
            let (un, vn) = (u.clone(), v.clone());
            let b = build_up_b(rho, dt, dx, dy, &u, &v);
            pressure_poisson_periodic(nit, &mut p, dx, dy, &b);
            for i in interior(h) {
                for j in 0..w {
                    let (east, west) = east_west(j, w);
                    let (uc, vc) = (un[(i, j)], vn[(i, j)]);
                    u[(i, j)] = uc
                        - uc * dt / dx * (uc - un[(i, west)])
                        - vc * dt / dy * (uc - un[(i - 1, j)])
                        - dt_2rho_dx * (p[(i, east)] - p[(i, west)])
                        + nu * (dt_dx2 * (un[(i, east)] - uc * 2 + un[(i, west)])
                            + dt_dy2 * (un[(i + 1, j)] - uc * 2 + un[(i - 1, j)]))
                        + force_dt;
                    v[(i, j)] = vc
                        - uc * dt / dx * (vc - vn[(i, west)])
                        - vc * dt / dy * (vc - vn[(i - 1, j)])
                        - dt_2rho_dy * (p[(i + 1, j)] - p[(i - 1, j)])
                        + nu * (dt_dx2 * (vn[(i, east)] - vc * 2 + vn[(i, west)])
                            + dt_dy2 * (vn[(i + 1, j)] - vc * 2 + vn[(i - 1, j)]));
                }
            }
            let last = np_index(-1, h);
            for a in [&mut u, &mut v] {
                a.row_mut(0).fill(Q32::ZERO);
                a.row_mut(last).fill(Q32::ZERO);
            }
            // NumPy gives inf or nan when sum(u) is 0; the Q32 division panics.
            let sum_u: Q32 = u.sum();
            udiff = (sum_u - un.sum()) / sum_u;
            stepcount += 1;
        }
        stepcount
    }
}
