//! Pythran shallow_water: a Lax-Wendroff step of the shallow-water equations,
//! run `I` times on an `H x W` grid of ones with a bump of 6. The kernel
//! writes most divisions as `//`, NumPy's floor division of floats, so
//! `dt // (2 * dx)`, `dt // dx` and `dt // dy` are 0.0 and `g // 2` is 4.0:
//! the interior updates subtract zero, and only the boundary copies change
//! `H`. The program still computes every term as written.
//!
//! ```python
//! def model(height, width, dtype):
//!     m = np.ones((height, width), dtype=dtype)
//!     m[height // 4, width // 4] = 6.0
//!     return m
//!
//! def step(H, U, V, dt=0.02, dx=1.0, dy=1.0):
//!     g = 9.80665
//!     H[:, 0] = H[:, 1]
//!     U[:, 0] = U[:, 1]
//!     V[:, 0] = -V[:, 1]
//!     H[:, -1] = H[:, -2]
//!     U[:, -1] = U[:, -2]
//!     V[:, -1] = -V[:, -2]
//!     H[0, :] = H[1, :]
//!     U[0, :] = -U[1, :]
//!     V[0, :] = V[1, :]
//!     H[-1, :] = H[-2, :]
//!     U[-1, :] = -U[-2, :]
//!     V[-1, :] = V[-2, :]
//!     Hx = (H[1:, 1:-1] + H[:-1, 1:-1]) // 2 - dt // (2 * dx) * (U[1:, 1:-1] - U[:-1, 1:-1])
//!     Ux = (U[1:, 1:-1] + U[:-1, 1:-1]) // 2 - dt / (2 * dx) * (U[1:, 1:-1] ** 2 // H[1:, 1:-1] + g // 2 * H[1:, 1:-1] ** 2 - (U[:-1, 1:-1] ** 2 // H[:-1, 1:-1] + g // 2 * H[:-1, 1:-1] ** 2))
//!     Vx = (V[1:, 1:-1] + V[:-1, 1:-1]) // 2 - dt // (2 * dx) * (U[1:, 1:-1] * V[1:, 1:-1] // H[1:, 1:-1] - U[:-1, 1:-1] * V[:-1, 1:-1] // H[:-1, 1:-1])
//!     Hy = (H[1:-1, 1:] + H[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] - V[1:-1, :-1])
//!     Uy = (U[1:-1, 1:] + U[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] * U[1:-1, 1:] // H[1:-1, 1:] - V[1:-1, :-1] * U[1:-1, :-1] // H[1:-1, :-1])
//!     Vy = (V[1:-1, 1:] + V[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] ** 2 // H[1:-1, 1:] + g // 2 * H[1:-1, 1:] ** 2 - (V[1:-1, :-1] ** 2 // H[1:-1, :-1] + g // 2 * H[1:-1, :-1] ** 2))
//!     H[1:-1, 1:-1] -= dt // dx * (Ux[1:, :] - Ux[:-1, :]) + dt // dy * (Vy[:, 1:] - Vy[:, :-1])
//!     U[1:-1, 1:-1] -= dt // dx * (Ux[1:, :] ** 2 // Hx[1:, :] + g // 2 * Hx[1:, :] ** 2 - (Ux[:-1, :] ** 2 // Hx[:-1, :] + g // 2 * Hx[:-1, :] ** 2)) + dt // dy * (Vy[:, 1:] * Uy[:, 1:] // Hy[:, 1:] - Vy[:, :-1] * Uy[:, :-1] // Hy[:, :-1])
//!     V[1:-1, 1:-1] -= dt // dx * (Ux[1:, :] * Vx[1:, :] // Hx[1:, :] - Ux[:-1, :] * Vx[:-1, :] // Hx[:-1, :]) + dt // dy * (Vy[:, 1:] ** 2 // Hy[:, 1:] + g // 2 * Hy[:, 1:] ** 2 - (Vy[:, :-1] ** 2 // Hy[:, :-1] + g // 2 * Hy[:, :-1] ** 2))
//!     return (H, U, V)
//!
//! def simulate(H, timesteps):
//!     U = np.zeros_like(H)
//!     V = np.zeros_like(H)
//!     for i in range(timesteps):
//!         H, U, V = step(H, U, V)
//!     return H
//!
//! def run(H: Integer, W: Integer, I: Integer):
//!     m = model(H, W, dtype=np.float64)
//!     m = simulate(m, I)
//!     return m
//! ```
use crate::prelude::*;

/// NumPy's `a // b` on floats: `floor(a / b)`, exact on the Q32 encodings.
fn fdiv(a: Q32, b: Q32) -> Q32 {
    Q32::from_int(floor_div(a.to_bits(), b.to_bits()))
}

/// `step`'s scalars: its defaults `dt = 0.02`, `dx = dy = 1.0` and
/// `g = 9.80665`, and the coefficients the kernel derives from them.
struct Coefficients {
    /// `dt // (2 * dx)` and `dt // (2 * dy)`.
    half_step_x: Q32,
    half_step_y: Q32,
    /// `dt / (2 * dx)`.
    half_ratio_x: Q32,
    /// `dt // dx` and `dt // dy`.
    step_x: Q32,
    step_y: Q32,
    /// `g // 2`.
    half_g: Q32,
}

impl Coefficients {
    fn new() -> Coefficients {
        let dt = Q32::from_bits(85_899_346); // round(0.02 * 2^32)
        let (dx, dy) = (Q32::ONE, Q32::ONE);
        let g = Q32::from_bits(42_119_241_033); // round(9.80665 * 2^32)
        Coefficients {
            half_step_x: fdiv(dt, dx * 2),
            half_step_y: fdiv(dt, dy * 2),
            half_ratio_x: dt / (dx * 2),
            step_x: fdiv(dt, dx),
            step_y: fdiv(dt, dy),
            half_g: fdiv(g, Q32::from_int(2)),
        }
    }

    /// `u ** 2 // h + g // 2 * h ** 2`, the flux term of the kernel.
    fn flux(&self, u: Q32, h: Q32) -> Q32 {
        fdiv(u.powi(2), h) + self.half_g * h.powi(2)
    }
}

/// The reflecting boundary of one field, in the kernel's order:
/// `a[:, 0] = ±a[:, 1]`, `a[:, -1] = ±a[:, -2]` (negated if
/// `negate_columns`), then `a[0, :] = ±a[1, :]`, `a[-1, :] = ±a[-2, :]`
/// (negated if `negate_rows`).
fn reflect(a: &mut Array2<Q32>, negate_columns: bool, negate_rows: bool) {
    let (h, w) = a.dim();
    let col_sign = |v: Q32| if negate_columns { -v } else { v };
    let row_sign = |v: Q32| if negate_rows { -v } else { v };
    for i in 0..h {
        a[(i, 0)] = col_sign(a[(i, 1)]);
        a[(i, w - 1)] = col_sign(a[(i, w - 2)]);
    }
    for j in 0..w {
        a[(0, j)] = row_sign(a[(1, j)]);
        a[(h - 1, j)] = row_sign(a[(h - 2, j)]);
    }
}

/// One `step` on `(H, U, V)`, in place.
fn step(c: &Coefficients, hh: &mut Array2<Q32>, uu: &mut Array2<Q32>, vv: &mut Array2<Q32>) {
    reflect(hh, false, false);
    reflect(uu, false, true);
    reflect(vv, true, false);
    let (h, w) = hh.dim();
    let two = Q32::from_int(2);

    // Half steps in x, on the faces between rows i and i + 1 (columns 1..w-1).
    let x_shape = (h - 1, w - 2);
    let hx = Array2::from_shape_fn(x_shape, |(i, j)| {
        let (a, b) = ((i + 1, j + 1), (i, j + 1));
        fdiv(hh[a] + hh[b], two) - c.half_step_x * (uu[a] - uu[b])
    });
    let ux = Array2::from_shape_fn(x_shape, |(i, j)| {
        let (a, b) = ((i + 1, j + 1), (i, j + 1));
        fdiv(uu[a] + uu[b], two) - c.half_ratio_x * (c.flux(uu[a], hh[a]) - c.flux(uu[b], hh[b]))
    });
    let vx = Array2::from_shape_fn(x_shape, |(i, j)| {
        let (a, b) = ((i + 1, j + 1), (i, j + 1));
        fdiv(vv[a] + vv[b], two)
            - c.half_step_x * (fdiv(uu[a] * vv[a], hh[a]) - fdiv(uu[b] * vv[b], hh[b]))
    });

    // Half steps in y, on the faces between columns j and j + 1 (rows 1..h-1).
    let y_shape = (h - 2, w - 1);
    let hy = Array2::from_shape_fn(y_shape, |(i, j)| {
        let (a, b) = ((i + 1, j + 1), (i + 1, j));
        fdiv(hh[a] + hh[b], two) - c.half_step_y * (vv[a] - vv[b])
    });
    let uy = Array2::from_shape_fn(y_shape, |(i, j)| {
        let (a, b) = ((i + 1, j + 1), (i + 1, j));
        fdiv(uu[a] + uu[b], two)
            - c.half_step_y * (fdiv(vv[a] * uu[a], hh[a]) - fdiv(vv[b] * uu[b], hh[b]))
    });
    let vy = Array2::from_shape_fn(y_shape, |(i, j)| {
        let (a, b) = ((i + 1, j + 1), (i + 1, j));
        fdiv(vv[a] + vv[b], two) - c.half_step_y * (c.flux(vv[a], hh[a]) - c.flux(vv[b], hh[b]))
    });

    // Full steps on the interior; each update reads only the half steps.
    for i in 0..h - 2 {
        for j in 0..w - 2 {
            let (xa, xb, ya, yb) = ((i + 1, j), (i, j), (i, j + 1), (i, j));
            hh[(i + 1, j + 1)] -= c.step_x * (ux[xa] - ux[xb]) + c.step_y * (vy[ya] - vy[yb]);
            uu[(i + 1, j + 1)] -= c.step_x * (c.flux(ux[xa], hx[xa]) - c.flux(ux[xb], hx[xb]))
                + c.step_y * (fdiv(vy[ya] * uy[ya], hy[ya]) - fdiv(vy[yb] * uy[yb], hy[yb]));
            vv[(i + 1, j + 1)] -= c.step_x
                * (fdiv(ux[xa] * vx[xa], hx[xa]) - fdiv(ux[xb] * vx[xb], hx[xb]))
                + c.step_y * (c.flux(vy[ya], hy[ya]) - c.flux(vy[yb], hy[yb]));
        }
    }
}

pub mod run {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let height: i64 = args.scalar();
        let width: i64 = args.scalar();
        let timesteps: i64 = args.scalar();
        let dim = |d: i64| usize::try_from(d).expect("negative dimensions are not allowed");
        let (h, w) = (dim(height), dim(width));

        // model(H, W)
        let mut hh = Array2::from_elem((h, w), Q32::ONE);
        hh[(np_index(floor_div(height, 4), h), np_index(floor_div(width, 4), w))] = Q32::from_int(6);

        // simulate(m, I)
        let coefficients = Coefficients::new();
        let mut uu = Array2::zeros((h, w));
        let mut vv = Array2::zeros((h, w));
        for _ in 0..timesteps {
            step(&coefficients, &mut hh, &mut uu, &mut vv);
        }
        hh
    }
}
