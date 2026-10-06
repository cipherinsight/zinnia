//! Pythran collatz: among the starting values `3..=n_max`, the one whose
//! Collatz sequence is longest, and that length (counting both ends).
//!
//! ```python
//! def collatz(n_max: Integer):
//!     n = np.arange(3, n_max + 1)
//!     count = np.ones_like(n)
//!     indices = n != 1
//!     while np.any(indices):
//!         count[indices] += 1
//!         x = n[indices]
//!         n[indices] = np.where(x & 1, 3 * x + 1, x // 2)
//!         indices = n != 1
//!     max_i = np.argmax(count)
//!     return (max_i + 3, count[max_i])
//! ```
//!
//! The masked loop advances every value that has not reached 1 yet, so each
//! `count` entry is 1 plus the number of steps its value takes to reach 1;
//! the program runs each sequence on its own.
use crate::prelude::*;

/// 1 plus the number of Collatz steps from `n` (here at least 3) to 1.
fn sequence_length(mut n: i64) -> i64 {
    let mut count = 1;
    while n != 1 {
        count += 1;
        n = if n & 1 != 0 { 3 * n + 1 } else { n / 2 };
    }
    count
}

pub mod collatz {
    use super::*;

    pub type Output = (i64, i64);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n_max: i64 = args.scalar();
        assert!((0..=1024).contains(&n_max), "requires 0 <= n_max <= 1024");
        let count: Vec<i64> = (3..=n_max).map(sequence_length).collect();
        // `np.argmax` of an empty array (n_max < 3) raises.
        let max_i = np::argmax(&count);
        (max_i as i64 + 3, count[max_i])
    }
}
