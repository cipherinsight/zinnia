//! Pythran fdtd: a leapfrog finite-difference time-domain wave update,
//! `steps` times.
//!
//! ```python
//! def fdtd(input_grid: Float[Array, '1000 1000'], steps: Integer):
//!     grid = input_grid.copy()
//!     old_grid = np.zeros_like(input_grid)
//!     previous_grid = np.zeros_like(input_grid)
//!     l_x = grid.shape[0]
//!     l_y = grid.shape[1]
//!     for i in range(steps):
//!         np.copyto(previous_grid, old_grid)
//!         np.copyto(old_grid, grid)
//!         for x in range(l_x):
//!             for y in range(l_y):
//!                 grid[x, y] = 0.0
//!                 if 0 < x + 1 < l_x:
//!                     grid[x, y] += old_grid[x + 1, y]
//!                 if 0 < x - 1 < l_x:
//!                     grid[x, y] += old_grid[x - 1, y]
//!                 if 0 < y + 1 < l_y:
//!                     grid[x, y] += old_grid[x, y + 1]
//!                 if 0 < y - 1 < l_y:
//!                     grid[x, y] += old_grid[x, y - 1]
//!                 grid[x, y] /= 2.0
//!                 grid[x, y] -= previous_grid[x, y]
//!     return grid
//! ```
//!
//! The guards are kept as written: `0 < x - 1` leaves out the neighbour in
//! row 0 (and column 0), unlike the neighbour in the last row.
use crate::prelude::*;

pub mod fdtd {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut grid: Array2<Q32> = args.array2();
        let steps: i64 = args.scalar();
        let (lx, ly) = grid.dim();
        let mut old = Array2::zeros((lx, ly));
        let mut previous = Array2::zeros((lx, ly));
        for _ in 0..steps {
            // previous <- old, old <- grid; grid is overwritten below.
            core::mem::swap(&mut previous, &mut old);
            core::mem::swap(&mut old, &mut grid);
            for x in 0..lx {
                for y in 0..ly {
                    let mut value = Q32::ZERO;
                    if x + 1 < lx {
                        value += old[(x + 1, y)];
                    }
                    if x >= 2 {
                        value += old[(x - 1, y)];
                    }
                    if y + 1 < ly {
                        value += old[(x, y + 1)];
                    }
                    if y >= 2 {
                        value += old[(x, y - 1)];
                    }
                    grid[(x, y)] = value / 2 - previous[(x, y)];
                }
            }
        }
        grid
    }
}
