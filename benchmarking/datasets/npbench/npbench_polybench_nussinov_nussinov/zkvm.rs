//! NPBench polybench/nussinov: Nussinov's dynamic program for the maximum
//! number of base pairs (`b1 + b2 == 3`) in an RNA sequence.
//!
//! ```python
//! def match(b1, b2):
//!     if b1 + b2 == 3:
//!         return 1
//!     else:
//!         return 0
//!
//! def kernel(N: Integer, seq: Integer[Array, '40']):
//!     table = np.zeros((N, N), np.int32)
//!     for i in range(N - 1, -1, -1):
//!         for j in range(i + 1, N):
//!             if j - 1 >= 0:
//!                 table[i, j] = max(table[i, j], table[i, j - 1])
//!             if i + 1 < N:
//!                 table[i, j] = max(table[i, j], table[i + 1, j])
//!             if j - 1 >= 0 and i + 1 < N:
//!                 if i < j - 1:
//!                     table[i, j] = max(table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j]))
//!                 else:
//!                     table[i, j] = max(table[i, j], table[i + 1, j - 1])
//!             for k in range(i + 1, j):
//!                 table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])
//!     return table
//! ```
//!
//! Inside the loops `0 <= i < j < N`, so `j - 1 >= 0` and `i + 1 < N` always
//! hold and those tests are left out. The table entries stay far below the
//! `int32` range (at most `N / 2`).
use crate::prelude::*;

/// The kernel's `match` (a Rust keyword).
fn match_bases(b1: i64, b2: i64) -> i64 {
    if b1 + b2 == 3 {
        1
    } else {
        0
    }
}

pub mod kernel {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let seq: Array1<i64> = args.array1();
        let n = usize::try_from(n).expect("negative dimensions are not allowed");
        let mut table = Array2::<i64>::zeros((n, n));
        for i in (0..n).rev() {
            for j in i + 1..n {
                let mut best = table[(i, j)].max(table[(i, j - 1)]).max(table[(i + 1, j)]);
                if i < j - 1 {
                    best = best.max(table[(i + 1, j - 1)] + match_bases(seq[i], seq[j]));
                } else {
                    best = best.max(table[(i + 1, j - 1)]);
                }
                for k in i + 1..j {
                    best = best.max(table[(i, k)] + table[(k + 1, j)]);
                }
                table[(i, j)] = best;
            }
        }
        table
    }
}
