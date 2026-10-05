//! Pythran fibo_seq: the Fibonacci sequence, iterated `n` times.
//!
//! ```python
//! @requires(lambda n: 0 <= n <= 1024)
//! def fibo(n: Integer):
//!     a, b = (1, 1)
//!     for _ in range(n):
//!         a, b = (a + b, a)
//!     return a
//! ```
use crate::prelude::*;

pub mod fibo {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let (mut a, mut b) = (1i64, 1i64);
        for _ in 0..n {
            // Wraps like int64 past n = 90 (Python's integers do not).
            (a, b) = (a.wrapping_add(b), a);
        }
        a
    }
}
