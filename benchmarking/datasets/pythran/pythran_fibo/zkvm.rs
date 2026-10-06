//! Pythran fibo: the `n`-th Fibonacci number by naive recursion (OpenMP tasks
//! above 10 in the original; sequential here).
//!
//! ```python
//! def rfibo(n):
//!     if n < 2:
//!         return n
//!     else:
//!         n_1 = rfibo(n - 1)
//!         n_2 = rfibo(n - 2)
//!         return n_1 + n_2
//!
//! def fibo(n):
//!     if n < 10:
//!         return rfibo(n)
//!     else:
//!         n_1 = 0
//!         'omp task shared(n,n_1)'
//!         n_1 = fibo(n - 1)
//!         n_2 = fibo(n - 2)
//!         'omp taskwait'
//!         return n_1 + n_2
//!
//! def test(n: Integer):
//!     """omp parallel"""
//!     'omp single'
//!     f = fibo(n)
//!     return f
//! ```
use crate::prelude::*;

fn rfibo(n: i64) -> i64 {
    if n < 2 {
        n
    } else {
        rfibo(n - 1).wrapping_add(rfibo(n - 2))
    }
}

/// The task-parallel recursion of the original, which only changes how the
/// same calls are scheduled.
fn fibo(n: i64) -> i64 {
    if n < 10 {
        rfibo(n)
    } else {
        fibo(n - 1).wrapping_add(fibo(n - 2))
    }
}

pub mod test {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        fibo(n)
    }
}
