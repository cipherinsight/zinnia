//! Pythran ising: one checkerboard Metropolis sweep of the Ising model on a
//! periodic grid of spins, at inverse temperature `beta`.
//!
//! ```python
//! def ising_step(field: Integer[Array, '2 2'], beta: Float=0.4):
//!     N, M = field.shape
//!     for n_offset in range(2):
//!         for m_offset in range(2):
//!             for n in range(n_offset, N, 2):
//!                 for m in range(m_offset, M, 2):
//!                     _ising_update(field, n, m, np.float32(beta))
//!     return field
//!
//! def clamp(v, n):
//!     if v < 0:
//!         return v + n
//!     elif v < n:
//!         return v
//!     else:
//!         return v - n
//!
//! def _ising_update(field, n, m, beta):
//!     total = 0
//!     N, M = field.shape
//!     for i in range(n - 1, n + 2):
//!         for j in range(m - 1, m + 2):
//!             if i == n and j == m:
//!                 continue
//!             total += field[clamp(i, N), clamp(j, M)]
//!     dE = 2 * field[n, m] * total
//!     if dE <= 0:
//!         field[n, m] *= -1
//!     elif np.exp(-dE * beta) > 0.5:
//!         field[n, m] *= -1
//! ```
//!
//! Zinnia has one float type, so `np.float32(beta)` is `beta` itself here
//! (NumPy rounds it to single precision first, e.g. 0.4 to 0.4000000059604645).
use crate::prelude::*;

const HALF: Q32 = Q32::from_bits(1 << 31); // 0.5

/// The neighbour of `n` at offset `d` in {-1, 0, 1} on a periodic axis of
/// length `len` (`clamp(n + d, len)`).
fn clamp(n: usize, d: isize, len: usize) -> usize {
    let v = n as isize + d;
    if v < 0 {
        (v + len as isize) as usize
    } else if (v as usize) < len {
        v as usize
    } else {
        v as usize - len
    }
}

fn ising_update(field: &mut Array2<i64>, n: usize, m: usize, beta: Q32) {
    let (rows, cols) = field.dim();
    let mut total = 0;
    for di in -1..=1 {
        for dj in -1..=1 {
            if di == 0 && dj == 0 {
                continue;
            }
            total += field[(clamp(n, di, rows), clamp(m, dj, cols))];
        }
    }
    let de = 2 * field[(n, m)] * total;
    if de <= 0 || (beta * -de).exp() > HALF {
        field[(n, m)] *= -1;
    }
}

pub mod ising_step {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut field: Array2<i64> = args.array2();
        let beta: Q32 = args.scalar();
        let (rows, cols) = field.dim();
        for n_offset in 0..2 {
            for m_offset in 0..2 {
                for n in (n_offset..rows).step_by(2) {
                    for m in (m_offset..cols).step_by(2) {
                        ising_update(&mut field, n, m, beta);
                    }
                }
            }
        }
        field
    }
}
