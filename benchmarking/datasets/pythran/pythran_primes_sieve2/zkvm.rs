//! Pythran primes_sieve2: the primes up to `n` by a sieve over the odd
//! numbers.
//!
//! ```python
//! @requires(lambda n: 0 <= n <= 1024)
//! def get_primes7(n: Integer):
//!     if n < 2:
//!         return []
//!     if n == 2:
//!         return [2]
//!     s = list(range(3, n + 1, 2))
//!     mroot = n ** 0.5
//!     half = len(s)
//!     i = 0
//!     m = 3
//!     while m <= mroot:
//!         if s[i]:
//!             j = (m * m - 3) // 2
//!             s[j] = 0
//!             while j < half:
//!                 s[j] = 0
//!                 j += m
//!         i = i + 1
//!         m = 2 * i + 3
//!     return [2] + [x for x in s if x]
//! ```
use crate::prelude::*;

pub mod get_primes7 {
    use super::*;

    /// The primes up to `n`, a list whose length depends on `n`.
    pub type Output = Vec<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        if n < 2 {
            return Vec::new();
        }
        if n == 2 {
            return vec![2];
        }
        // s[i] holds the odd number 2i + 3, or 0 once it is crossed out.
        let mut s: Vec<i64> = (3..=n).step_by(2).collect();
        let half = s.len();
        let (mut i, mut m) = (0usize, 3usize);
        // `m <= n ** 0.5` for a non-negative integer m is `m * m <= n`.
        while (m * m) as i64 <= n {
            if s[i] != 0 {
                // The kernel's `s[j] = 0` before the loop is the loop's first
                // step: j = (m^2 - 3) / 2 < half because m^2 <= n.
                let mut j = (m * m - 3) / 2;
                while j < half {
                    s[j] = 0;
                    j += m;
                }
            }
            i += 1;
            m = 2 * i + 3;
        }
        core::iter::once(2).chain(s.into_iter().filter(|&x| x != 0)).collect()
    }
}
