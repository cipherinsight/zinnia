//! Pythran growcut: one GrowCut cellular-automaton step on an `N x N`
//! all-black image whose only seed is cell `(0, 0)`; returns how many times a
//! cell was conquered.
//!
//! ```python
//! def window_floor(idx, radius):
//!     if radius > idx:
//!         return 0
//!     else:
//!         return idx - radius
//!
//! def window_ceil(idx, ceil, radius):
//!     if idx + radius > ceil:
//!         return ceil
//!     else:
//!         return idx + radius
//!
//! def python_kernel(image, state, state_next, window_radius):
//!     changes = 0
//!     sqrt_3 = math.sqrt(3.0)
//!     height = image.shape[0]
//!     width = image.shape[1]
//!     for j in range(width):
//!         for i in range(height):
//!             winning_colony = state[i, j, 0]
//!             defense_strength = state[i, j, 1]
//!             for jj in range(window_floor(j, window_radius), window_ceil(j + 1, width, window_radius)):
//!                 for ii in range(window_floor(i, window_radius), window_ceil(i + 1, height, window_radius)):
//!                     if ii == i and jj == j:
//!                         continue
//!                     d = image[i, j, 0] - image[ii, jj, 0]
//!                     s = d * d
//!                     for k in range(1, 3):
//!                         d = image[i, j, k] - image[ii, jj, k]
//!                         s += d * d
//!                     gval = 1.0 - math.sqrt(s) / sqrt_3
//!                     attack_strength = gval * state[ii, jj, 1]
//!                     if attack_strength > defense_strength:
//!                         defense_strength = attack_strength
//!                         winning_colony = state[ii, jj, 0]
//!                         changes += 1
//!             state_next[i, j, 0] = winning_colony
//!             state_next[i, j, 1] = defense_strength
//!     return changes
//!
//! def test(N: Integer):
//!     image = np.zeros((N, N, 3))
//!     state = np.zeros((N, N, 2))
//!     state_next = np.empty_like(state)
//!     state[0, 0, 0] = 1
//!     state[0, 0, 1] = 1
//!     return python_kernel(image, state, state_next, 10)
//! ```
use crate::prelude::*;

/// `math.sqrt(3.0)`, quantized.
const SQRT_3: Q32 = Q32::from_bits(7_439_101_574);

fn window_floor(idx: usize, radius: usize) -> usize {
    idx.saturating_sub(radius)
}

fn window_ceil(idx: usize, ceil: usize, radius: usize) -> usize {
    (idx + radius).min(ceil)
}

/// `python_kernel`'s count of conquests. `test` reads neither `state_next`
/// nor the colony labels (`state[:, :, 0]`), so the step keeps only the
/// defense strengths `strength = state[:, :, 1]`.
fn changes(image: &Array3<Q32>, strength: &Array2<Q32>, window_radius: usize) -> i64 {
    let (height, width, _) = image.dim();
    let mut changes = 0;
    for j in 0..width {
        for i in 0..height {
            let mut defense_strength = strength[(i, j)];
            for jj in window_floor(j, window_radius)..window_ceil(j + 1, width, window_radius) {
                for ii in window_floor(i, window_radius)..window_ceil(i + 1, height, window_radius)
                {
                    if ii == i && jj == j {
                        continue;
                    }
                    let mut s = Q32::ZERO;
                    for k in 0..3 {
                        let d = image[(i, j, k)] - image[(ii, jj, k)];
                        s += d * d;
                    }
                    let gval = Q32::ONE - s.sqrt() / SQRT_3;
                    let attack_strength = gval * strength[(ii, jj)];
                    if attack_strength > defense_strength {
                        defense_strength = attack_strength;
                        changes += 1;
                    }
                }
            }
        }
    }
    changes
}

pub mod test {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= N <= 1024");
        let n = n as usize;
        let image = Array3::<Q32>::zeros((n, n, 3));
        let mut strength = Array2::<Q32>::zeros((n, n));
        strength[(0, 0)] = Q32::ONE;
        changes(&image, &strength, 10)
    }
}
