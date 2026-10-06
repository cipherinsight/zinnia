//! Pythran d2q9_nxnyns: one lattice-Boltzmann (D2Q9, multiple-relaxation-
//! time) step: periodic boundaries on `f1`, then for every interior node
//! stream from `f1`, relax in moment space and store into `f2`.
//!
//! ```python
//! def m2f_loc(m, f):
//!     c0 = 1.0 / 6
//!     c1 = 1.0 / 9
//!     c2 = 1.0 / 18
//!     c3 = 1.0 / 36
//!     c4 = 1.0 / 12
//!     f[0] = c1 * m[0] - c1 * m[3] + c1 * m[4]
//!     f[1] = c1 * m[0] + c0 * m[1] - c3 * m[3] - c2 * m[4] - c0 * m[5] + 0.25 * m[7]
//!     f[2] = c1 * m[0] + c0 * m[2] - c3 * m[3] - c2 * m[4] - c0 * m[6] - 0.25 * m[7]
//!     f[3] = c1 * m[0] - c0 * m[1] - c3 * m[3] - c2 * m[4] + c0 * m[5] + 0.25 * m[7]
//!     f[4] = c1 * m[0] - c0 * m[2] - c3 * m[3] - c2 * m[4] + c0 * m[6] - 0.25 * m[7]
//!     f[5] = c1 * m[0] + c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] + c4 * m[6] + 0.25 * m[8]
//!     f[6] = c1 * m[0] - c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] + c4 * m[6] - 0.25 * m[8]
//!     f[7] = c1 * m[0] - c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] - c4 * m[6] + 0.25 * m[8]
//!     f[8] = c1 * m[0] + c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] - c4 * m[6] - 0.25 * m[8]
//!
//! def f2m_loc(f, m):
//!     m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
//!     m[1] = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
//!     m[2] = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
//!     m[3] = -4.0 * f[0] - f[1] - f[2] - f[3] - f[4] + 2.0 * f[5] + 2.0 * f[6] + 2.0 * f[7] + 2.0 * f[8]
//!     m[4] = 4.0 * f[0] - 2.0 * f[1] - 2.0 * f[2] - 2.0 * f[3] - 2.0 * f[4] + f[5] + f[6] + f[7] + f[8]
//!     m[5] = -2.0 * f[1] + 2.0 * f[3] + f[5] - f[6] - f[7] + f[8]
//!     m[6] = -2.0 * f[2] + 2.0 * f[4] + f[5] + f[6] - f[7] - f[8]
//!     m[7] = f[1] - f[2] + f[3] - f[4]
//!     m[8] = f[5] - f[6] + f[7] - f[8]
//!
//! def getf(f, floc, i, j):
//!     floc[0] = f[i, j, 0]
//!     floc[1] = f[i - 1, j, 1]
//!     floc[2] = f[i, j - 1, 2]
//!     floc[3] = f[i + 1, j, 3]
//!     floc[4] = f[i, j + 1, 4]
//!     floc[5] = f[i - 1, j - 1, 5]
//!     floc[6] = f[i + 1, j - 1, 6]
//!     floc[7] = f[i + 1, j + 1, 7]
//!     floc[8] = f[i - 1, j + 1, 8]
//!
//! def setf(f, floc, i, j):
//!     f[j, i, 0] = floc[0]
//!     ...                      (f[j, i, k] = floc[k] for k = 0..8)
//!     f[j, i, 8] = floc[8]
//!
//! def relaxation_loc(m):
//!     m[3] += 1.1312217194570136 * (-2 * m[0] + 3.0 * m[1] * m[1] + 3.0 * m[2] * m[2] - m[3])
//!     m[4] += 1.1312217194570136 * (m[0] + 1.5 * m[1] * m[1] + 1.5 * m[2] * m[2] - m[4])
//!     m[5] += 1.1312217194570136 * (-m[1] - m[5])
//!     m[6] += 1.1312217194570136 * (-m[2] - m[6])
//!     m[7] += 1.8573551263001487 * (m[1] * m[1] - m[2] * m[2] - m[7])
//!     m[8] += 1.8573551263001487 * (m[1] * m[2] - m[8])
//!
//! def periodic_bc(f):
//!     nx, ny, ns = f.shape
//!     for j in range(ny):
//!         for k in range(ns):
//!             f[0, j, k] = f[nx - 2, j, k]
//!             f[nx - 1, j, k] = f[1, j, k]
//!     for i in range(nx):
//!         for k in range(ns):
//!             f[i, 0, k] = f[i, ny - 2, k]
//!             f[i, ny - 1, k] = f[i, 1, k]
//!
//! def one_time_step(f1: Float[Array, '50 50 9'], f2: Float[Array, '50 50 9']):
//!     nx, ny, ns = f1.shape
//!     floc = np.zeros(ns)
//!     mloc = np.zeros(ns)
//!     periodic_bc(f1)
//!     for i in range(1, nx - 1):
//!         for j in range(1, ny - 1):
//!             getf(f1, floc, i, j)
//!             f2m_loc(floc, mloc)
//!             relaxation_loc(mloc)
//!             m2f_loc(mloc, floc)
//!             setf(f2, floc, i, j)
//!     return f2
//! ```
use crate::prelude::*;

/// Python's `1.0 / 6`, `1.0 / 9`, `1.0 / 18`, `1.0 / 36`, `1.0 / 12`, quantized.
const C0: Q32 = Q32::from_bits(715_827_883);
const C1: Q32 = Q32::from_bits(477_218_588);
const C2: Q32 = Q32::from_bits(238_609_294);
const C3: Q32 = Q32::from_bits(119_304_647);
const C4: Q32 = Q32::from_bits(357_913_941);
const QUARTER: Q32 = Q32::from_bits(1 << 30); // 0.25
const THREE_HALVES: Q32 = Q32::from_bits(3 << 31); // 1.5
/// The relaxation rates `1.1312217194570136` and `1.8573551263001487`, quantized.
const OMEGA_E: Q32 = Q32::from_bits(4_858_560_290);
const OMEGA_XY: Q32 = Q32::from_bits(7_977_279_525);

/// The nine populations or moments of one node.
type Node = [Q32; 9];

fn m2f_loc(m: &Node) -> Node {
    [
        C1 * m[0] - C1 * m[3] + C1 * m[4],
        C1 * m[0] + C0 * m[1] - C3 * m[3] - C2 * m[4] - C0 * m[5] + QUARTER * m[7],
        C1 * m[0] + C0 * m[2] - C3 * m[3] - C2 * m[4] - C0 * m[6] - QUARTER * m[7],
        C1 * m[0] - C0 * m[1] - C3 * m[3] - C2 * m[4] + C0 * m[5] + QUARTER * m[7],
        C1 * m[0] - C0 * m[2] - C3 * m[3] - C2 * m[4] + C0 * m[6] - QUARTER * m[7],
        C1 * m[0]
            + C0 * m[1]
            + C0 * m[2]
            + C2 * m[3]
            + C3 * m[4]
            + C4 * m[5]
            + C4 * m[6]
            + QUARTER * m[8],
        C1 * m[0] - C0 * m[1] + C0 * m[2] + C2 * m[3] + C3 * m[4] - C4 * m[5] + C4 * m[6]
            - QUARTER * m[8],
        C1 * m[0] - C0 * m[1] - C0 * m[2] + C2 * m[3] + C3 * m[4] - C4 * m[5] - C4 * m[6]
            + QUARTER * m[8],
        C1 * m[0] + C0 * m[1] - C0 * m[2] + C2 * m[3] + C3 * m[4] + C4 * m[5]
            - C4 * m[6]
            - QUARTER * m[8],
    ]
}

/// The integer-valued float factors (`4.0`, `2.0`) multiply exactly.
fn f2m_loc(f: &Node) -> Node {
    [
        f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8],
        f[1] - f[3] + f[5] - f[6] - f[7] + f[8],
        f[2] - f[4] + f[5] + f[6] - f[7] - f[8],
        f[0] * -4 - f[1] - f[2] - f[3] - f[4] + f[5] * 2 + f[6] * 2 + f[7] * 2 + f[8] * 2,
        f[0] * 4 - f[1] * 2 - f[2] * 2 - f[3] * 2 - f[4] * 2 + f[5] + f[6] + f[7] + f[8],
        f[1] * -2 + f[3] * 2 + f[5] - f[6] - f[7] + f[8],
        f[2] * -2 + f[4] * 2 + f[5] + f[6] - f[7] - f[8],
        f[1] - f[2] + f[3] - f[4],
        f[5] - f[6] + f[7] - f[8],
    ]
}

fn relaxation_loc(m: &mut Node) {
    m[3] += OMEGA_E * (m[0] * -2 + m[1] * 3 * m[1] + m[2] * 3 * m[2] - m[3]);
    m[4] += OMEGA_E * (m[0] + THREE_HALVES * m[1] * m[1] + THREE_HALVES * m[2] * m[2] - m[4]);
    m[5] += OMEGA_E * (-m[1] - m[5]);
    m[6] += OMEGA_E * (-m[2] - m[6]);
    m[7] += OMEGA_XY * (m[1] * m[1] - m[2] * m[2] - m[7]);
    m[8] += OMEGA_XY * (m[1] * m[2] - m[8]);
}

/// Streaming: population `k` of node `(i, j)` comes from the neighbour it
/// moves away from.
fn getf(f: &Array3<Q32>, i: usize, j: usize) -> Node {
    [
        f[(i, j, 0)],
        f[(i - 1, j, 1)],
        f[(i, j - 1, 2)],
        f[(i + 1, j, 3)],
        f[(i, j + 1, 4)],
        f[(i - 1, j - 1, 5)],
        f[(i + 1, j - 1, 6)],
        f[(i + 1, j + 1, 7)],
        f[(i - 1, j + 1, 8)],
    ]
}

/// Copies the second and the second-to-last plane of each axis onto the
/// opposite edge. `nx - 2` and `ny - 2` wrap like NumPy indices (and, like
/// every index here, are only checked when a loop body runs).
fn periodic_bc(f: &mut Array3<Q32>) {
    let (nx, ny, ns) = f.dim();
    for j in 0..ny {
        for k in 0..ns {
            f[(0, j, k)] = f[(np_index(nx as i64 - 2, nx), j, k)];
            f[(np_index(nx as i64 - 1, nx), j, k)] = f[(1, j, k)];
        }
    }
    for i in 0..nx {
        for k in 0..ns {
            f[(i, 0, k)] = f[(i, np_index(ny as i64 - 2, ny), k)];
            f[(i, np_index(ny as i64 - 1, ny), k)] = f[(i, 1, k)];
        }
    }
}

pub mod one_time_step {
    use super::*;

    pub type Output = Array3<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut f1: Array3<Q32> = args.array3();
        let mut f2: Array3<Q32> = args.array3();
        let (nx, ny, _) = f1.dim();
        periodic_bc(&mut f1);
        for i in 1..nx.saturating_sub(1) {
            for j in 1..ny.saturating_sub(1) {
                let mut m = f2m_loc(&getf(&f1, i, j));
                relaxation_loc(&mut m);
                // `setf` stores node (i, j) at f2[j, i].
                for (k, value) in m2f_loc(&m).into_iter().enumerate() {
                    f2[(j, i, k)] = value;
                }
            }
        }
        f2
    }
}
