//! Pythran nqueens: every placement of `queen_count` non-attacking queens,
//! found by filtering all permutations of the columns.
//!
//! ```python
//! def permutations(iterable, r=None):
//!     pool = tuple(iterable)
//!     n = len(pool)
//!     if r is None:
//!         r = n
//!     indices = list(range(n))
//!     cycles = list(range(n - r + 1, n + 1))[::-1]
//!     yield tuple((pool[i] for i in indices[:r]))
//!     while n:
//!         for i in reversed(range(r)):
//!             cycles[i] -= 1
//!             if cycles[i] == 0:
//!                 indices[i:] = indices[i + 1:] + indices[i:i + 1]
//!                 cycles[i] = n - i
//!             else:
//!                 j = cycles[i]
//!                 indices[i], indices[-j] = (indices[-j], indices[i])
//!                 yield tuple((pool[i] for i in indices[:r]))
//!                 break
//!         else:
//!             return
//!
//! def n_queens(queen_count: Integer):
//!     out = list()
//!     cols = list(range(queen_count))
//!     for vec in permutations(cols, None):
//!         if queen_count == len(set((vec[i] + i for i in cols))) == len(set((vec[i] - i for i in cols))):
//!             out.append(vec)
//!     return out
//! ```
use crate::prelude::*;

/// The generator `permutations(range(n))` (`itertools.permutations`'
/// algorithm, `r = n`): calls `visit` with each permutation, in its order.
fn permutations(n: usize, mut visit: impl FnMut(&[usize])) {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut cycles: Vec<usize> = (1..=n).rev().collect();
    visit(&indices);
    'next: loop {
        for i in (0..n).rev() {
            cycles[i] -= 1;
            if cycles[i] == 0 {
                indices[i..].rotate_left(1);
                cycles[i] = n - i;
            } else {
                indices.swap(i, n - cycles[i]);
                visit(&indices);
                continue 'next;
            }
        }
        // The `for` loop ran out (`else: return`); for n = 0 it is empty,
        // which `while n:` also ends.
        return;
    }
}

/// Whether no two queens of `vec` (queen `i` in column `vec[i]`) share a
/// diagonal: the `n` values `vec[i] + i` are distinct, and so are the
/// `vec[i] - i`.
fn diagonals_free(vec: &[usize]) -> bool {
    (0..vec.len())
        .all(|i| (i + 1..vec.len()).all(|j| vec[i] + i != vec[j] + j && vec[i] + j != vec[j] + i))
}

pub mod n_queens {
    use super::*;

    /// One row per solution, as NumPy holds the list of tuples.
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let queen_count: i64 = args.scalar();
        // `range` of a negative count is empty; then `queen_count == 0` fails
        // for the one (empty) permutation.
        let n = usize::try_from(queen_count.max(0)).expect("queen_count exceeds the address space");
        let mut out = Vec::new();
        let mut solutions = 0;
        permutations(n, |vec| {
            if queen_count == n as i64 && diagonals_free(vec) {
                out.extend(vec.iter().map(|&c| c as i64));
                solutions += 1;
            }
        });
        Array2::from_shape_vec((solutions, n), out).expect("each solution has n columns")
    }
}
