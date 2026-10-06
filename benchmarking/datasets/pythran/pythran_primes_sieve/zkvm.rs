//! Pythran primes_sieve: the primes below `limit`, by the sieve of
//! Eratosthenes.
//!
//! ```python
//! def primes_sieve(limit: Integer):
//!     a = [True] * limit
//!     a[0] = a[1] = False
//!     primes = list()
//!     for i, isprime in enumerate(a):
//!         if isprime:
//!             primes.append(i)
//!             for n in range(i * i, limit, i):
//!                 a[n] = False
//!     return primes
//! ```
use crate::prelude::*;

pub mod primes_sieve {
    use super::*;

    pub type Output = Vec<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let limit: i64 = args.scalar();
        // `a[0] = a[1] = False` raises IndexError unless the list has two entries.
        assert!(limit >= 2, "list assignment index out of range");
        let limit = usize::try_from(limit).expect("limit exceeds the address space");
        let mut a = vec![true; limit];
        a[0] = false;
        a[1] = false;
        let mut primes = Vec::new();
        // `enumerate(a)` sees the entries crossed off while it runs.
        for i in 0..limit {
            if a[i] {
                primes.push(i as i64);
                // Saturating: `i * i` overflows a 32-bit usize (RISC Zero) past 65535.
                for n in (i.saturating_mul(i)..limit).step_by(i) {
                    a[n] = false;
                }
            }
        }
        primes
    }
}
