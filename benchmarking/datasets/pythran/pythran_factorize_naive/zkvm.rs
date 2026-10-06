//! Pythran factorize_naive: the prime factors of `n` (with multiplicity) by
//! trial division.
//!
//! ```python
//! def factorize_naive(n: Integer):
//!     if n < 2:
//!         return []
//!     factors = []
//!     p = 2
//!     while True:
//!         if n == 1:
//!             return factors
//!         r = n % p
//!         if r == 0:
//!             factors.append(p)
//!             n = n / p
//!         elif p * p >= n:
//!             factors.append(n)
//!             return factors
//!         elif p > 2:
//!             p += 2
//!         else:
//!             p += 1
//!     assert False, 'unreachable'
//! ```
//!
//! `n / p` is Python's true division, so `n` becomes a float after the first
//! factor; but `p` divides `n` (`r == 0`), so `n` stays a whole number and the
//! integer quotient is exact. The factors are output as integers (NumPy holds
//! the list as `float64` once a division happened; the values are equal).
use crate::prelude::*;

pub mod factorize_naive {
    use super::*;

    pub type Output = Vec<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let mut factors = Vec::new();
        if n < 2 {
            return factors;
        }
        let mut p = 2;
        loop {
            if n == 1 {
                return factors;
            }
            if n % p == 0 {
                factors.push(p);
                n /= p;
            } else if p * p >= n {
                factors.push(n);
                return factors;
            } else if p > 2 {
                p += 2;
            } else {
                p += 1;
            }
        }
    }
}
