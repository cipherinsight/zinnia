//! Pythran sexy_primes: the pairs of primes `(j - 6, j)` with `j <= x`, by
//! trial division.
//!
//! ```python
//! def is_prime(n):
//!     return all((n % j > 0 for j in range(2, n)))
//!
//! def primes_below(x: Integer):
//!     return [[j - 6, j] for j in range(9, x + 1) if is_prime(j) and is_prime(j - 6)]
//! ```
use crate::prelude::*;

/// Trial division by every `j` in `[2, n)`, stopping at the first divisor
/// (`all` short-circuits). Only called with `n >= 3`, so `%` is Python's.
fn is_prime(n: i64) -> bool {
    (2..n).all(|j| n % j > 0)
}

pub mod primes_below {
    use super::*;

    /// One row `[j - 6, j]` per pair; the number of rows depends on `x`.
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: i64 = args.scalar();
        let pairs: Vec<i64> = (9..=x)
            .filter(|&j| is_prime(j) && is_prime(j - 6))
            .flat_map(|j| [j - 6, j])
            .collect();
        Array2::from_shape_vec((pairs.len() / 2, 2), pairs).expect("two entries per pair")
    }
}
