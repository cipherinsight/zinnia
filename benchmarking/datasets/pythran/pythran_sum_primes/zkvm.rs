//! Pythran sum_primes: the sum of the primes below `n`, each found by trial
//! division.
//!
//! ```python
//! def isprime(n):
//!     if n < 2:
//!         return False
//!     if n == 2:
//!         return True
//!     max = int(math.ceil(math.sqrt(n)))
//!     i = 2
//!     while i <= max:
//!         if n % i == 0:
//!             return False
//!         i += 1
//!     return True
//!
//! @requires(lambda n: 0 <= n <= 1024)
//! def sum_primes(n: Integer):
//!     return sum([x for x in range(2, n) if isprime(x)])
//! ```
use crate::prelude::*;

/// `isprime(n)`: trial division by 2 ..= ceil(sqrt(n)).
fn isprime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    // `int(math.ceil(math.sqrt(n)))` from the exact integer root.
    let root = n.isqrt();
    let max = if root * root == n { root } else { root + 1 };
    (2..=max).all(|i| n % i != 0)
}

pub mod sum_primes {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        (2..n).filter(|&x| isprime(x)).sum()
    }
}
