//! NPBench polybench/adi: alternating-direction implicit solver of the 2-D
//! heat equation; each time step solves tridiagonal systems along the
//! columns (into `v`) and then along the rows (into `u`).
//!
//! ```python
//! @requires(lambda TSTEPS, N, u: 0 <= N <= 1024)
//! @requires(lambda TSTEPS, N, u: N != 0)
//! @requires(lambda TSTEPS, N, u: TSTEPS != 0)
//! def kernel(TSTEPS: Integer, N: Integer, u: Float[Array, '100 100']):
//!     v = np.empty(u.shape, dtype=u.dtype)
//!     p = np.empty(u.shape, dtype=u.dtype)
//!     q = np.empty(u.shape, dtype=u.dtype)
//!     DX = 1.0 / N
//!     DY = 1.0 / N
//!     DT = 1.0 / TSTEPS
//!     B1 = 2.0
//!     B2 = 1.0
//!     mul1 = B1 * DT / (DX * DX)
//!     mul2 = B2 * DT / (DY * DY)
//!     a = -mul1 / 2.0
//!     b = 1.0 + mul2
//!     c = a
//!     d = -mul2 / 2.0
//!     e = 1.0 + mul2
//!     f = d
//!     for t in range(1, TSTEPS + 1):
//!         v[0, 1:N - 1] = 1.0
//!         p[1:N - 1, 0] = 0.0
//!         q[1:N - 1, 0] = v[0, 1:N - 1]
//!         for j in range(1, N - 1):
//!             p[1:N - 1, j] = -c / (a * p[1:N - 1, j - 1] + b)
//!             q[1:N - 1, j] = (-d * u[j, 0:N - 2] + (1.0 + 2.0 * d) * u[j, 1:N - 1] - f * u[j, 2:N] - a * q[1:N - 1, j - 1]) / (a * p[1:N - 1, j - 1] + b)
//!         v[N - 1, 1:N - 1] = 1.0
//!         for j in range(N - 2, 0, -1):
//!             v[j, 1:N - 1] = p[1:N - 1, j] * v[j + 1, 1:N - 1] + q[1:N - 1, j]
//!         u[1:N - 1, 0] = 1.0
//!         p[1:N - 1, 0] = 0.0
//!         q[1:N - 1, 0] = u[1:N - 1, 0]
//!         for j in range(1, N - 1):
//!             p[1:N - 1, j] = -f / (d * p[1:N - 1, j - 1] + e)
//!             q[1:N - 1, j] = (-a * v[0:N - 2, j] + (1.0 + 2.0 * a) * v[1:N - 1, j] - c * v[2:N, j] - d * q[1:N - 1, j - 1]) / (d * p[1:N - 1, j - 1] + e)
//!         u[1:N - 1, N - 1] = 1.0
//!         for j in range(N - 2, 0, -1):
//!             u[1:N - 1, j] = p[1:N - 1, j] * u[1:N - 1, j + 1] + q[1:N - 1, j]
//!     return u
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let tsteps: i64 = args.scalar();
        let n: i64 = args.scalar();
        let mut u: Array2<Q32> = args.array2();
        assert!((0..=1024).contains(&n), "requires 0 <= N <= 1024");
        assert!(n != 0, "requires N != 0");
        assert!(tsteps != 0, "requires TSTEPS != 0");

        let dx = Q32::ONE / n;
        let dy = Q32::ONE / n;
        let dt = Q32::ONE / tsteps;
        let (b1, b2) = (Q32::from_int(2), Q32::ONE);
        let mul1 = b1 * dt / (dx * dx);
        let mul2 = b2 * dt / (dy * dy);
        let a = -mul1 / 2;
        let b = Q32::ONE + mul2;
        let c = a;
        let d = -mul2 / 2;
        let e = Q32::ONE + mul2;
        let f = d;
        // The loop-invariant factors `1.0 + 2.0 * d` and `1.0 + 2.0 * a`.
        let (centre_u, centre_v) = (Q32::ONE + d * 2, Q32::ONE + a * 2);

        let n = n as usize;
        // np.empty: every entry the kernel reads is written first.
        let mut v = Array2::<Q32>::zeros(u.raw_dim());
        let mut p = Array2::<Q32>::zeros(u.raw_dim());
        let mut q = Array2::<Q32>::zeros(u.raw_dim());
        let inner = 1..n - 1; // the slice 1:N-1
        for _ in 1..=tsteps {
            // Column sweep: v from u.
            for i in inner.clone() {
                v[(0, i)] = Q32::ONE;
                p[(i, 0)] = Q32::ZERO;
                q[(i, 0)] = v[(0, i)];
            }
            for j in inner.clone() {
                for i in inner.clone() {
                    let denominator = a * p[(i, j - 1)] + b;
                    p[(i, j)] = -c / denominator;
                    q[(i, j)] = (-d * u[(j, i - 1)] + centre_u * u[(j, i)]
                        - f * u[(j, i + 1)]
                        - a * q[(i, j - 1)])
                        / denominator;
                }
            }
            for i in inner.clone() {
                v[(n - 1, i)] = Q32::ONE;
            }
            for j in inner.clone().rev() {
                for i in inner.clone() {
                    v[(j, i)] = p[(i, j)] * v[(j + 1, i)] + q[(i, j)];
                }
            }
            // Row sweep: u from v.
            for i in inner.clone() {
                u[(i, 0)] = Q32::ONE;
                p[(i, 0)] = Q32::ZERO;
                q[(i, 0)] = u[(i, 0)];
            }
            for j in inner.clone() {
                for i in inner.clone() {
                    let denominator = d * p[(i, j - 1)] + e;
                    p[(i, j)] = -f / denominator;
                    q[(i, j)] = (-a * v[(i - 1, j)] + centre_v * v[(i, j)]
                        - c * v[(i + 1, j)]
                        - d * q[(i, j - 1)])
                        / denominator;
                }
            }
            for i in inner.clone() {
                u[(i, n - 1)] = Q32::ONE;
            }
            for j in inner.clone().rev() {
                for i in inner.clone() {
                    u[(i, j)] = p[(i, j)] * u[(i, j + 1)] + q[(i, j)];
                }
            }
        }
        u
    }
}
