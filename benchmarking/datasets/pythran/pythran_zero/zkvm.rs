//! Pythran zero: an `m x n` nested list of integer zeros.
//!
//! ```python
//! @requires(lambda n, m: 0 <= n <= 1024)
//! @requires(lambda n, m: 0 <= m <= 1024)
//! def zero(n: Integer, m: Integer):
//!     return [[0 for row in range(n)] for col in range(m)]
//! ```
use crate::prelude::*;

pub mod zero {
    use super::*;

    /// `m` rows of `n` zeros.
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let m: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        assert!((0..=1024).contains(&m), "requires 0 <= m <= 1024");
        Array2::zeros((m as usize, n as usize))
    }
}
