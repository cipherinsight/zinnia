//! Pythran queens_numba: place `n` mutually non-attacking queens on an 8 x 8
//! board by backtracking, trying the cells in row-major order at every step.
//!
//! ```python
//! def hits(x1, y1, x2, y2):
//!     return x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2)
//!
//! def hitsany(x, y, queens_x, queens_y):
//!     for i in range(len(queens_x)):
//!         if hits(x, y, queens_x[i], queens_y[i]):
//!             return True
//!     return False
//!
//! def _solve(n, queens_x, queens_y):
//!     if n == 0:
//!         return True
//!     for x in range(1, 9):
//!         for y in range(1, 9):
//!             if not hitsany(x, y, queens_x, queens_y):
//!                 queens_x.append(x)
//!                 queens_y.append(y)
//!                 if _solve(n - 1, queens_x, queens_y):
//!                     return True
//!                 queens_x.pop()
//!                 queens_y.pop()
//!     return False
//!
//! @requires(lambda n: 0 <= n <= 1024)
//! def solve(n: Integer):
//!     queens_x = []
//!     queens_y = []
//!     if _solve(n, queens_x, queens_y):
//!         return (queens_x, queens_y)
//!     else:
//!         return None
//! ```
use crate::prelude::*;

/// Whether a queen at (x1, y1) hits a queen at (x2, y2).
fn hits(x1: i64, y1: i64, x2: i64, y2: i64) -> bool {
    x1 == x2 || y1 == y2 || (x1 - x2).abs() == (y1 - y2).abs()
}

/// Whether a queen at (x, y) hits any of the queens placed so far.
fn hits_any(x: i64, y: i64, queens_x: &[i64], queens_y: &[i64]) -> bool {
    queens_x
        .iter()
        .zip(queens_y)
        .any(|(&qx, &qy)| hits(x, y, qx, qy))
}

/// `_solve`: places `n` more queens; on success the placements stay in
/// `queens_x` and `queens_y`.
fn place(n: i64, queens_x: &mut Vec<i64>, queens_y: &mut Vec<i64>) -> bool {
    if n == 0 {
        return true;
    }
    for x in 1..9 {
        for y in 1..9 {
            if !hits_any(x, y, queens_x, queens_y) {
                queens_x.push(x);
                queens_y.push(y);
                if place(n - 1, queens_x, queens_y) {
                    return true;
                }
                queens_x.pop();
                queens_y.pop();
            }
        }
    }
    false
}

pub mod solve {
    use super::*;

    /// `None` where the kernel returns None (no placement exists).
    pub type Output = Option<(Vec<i64>, Vec<i64>)>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let (mut queens_x, mut queens_y) = (Vec::new(), Vec::new());
        if place(n, &mut queens_x, &mut queens_y) {
            Some((queens_x, queens_y))
        } else {
            None
        }
    }
}
