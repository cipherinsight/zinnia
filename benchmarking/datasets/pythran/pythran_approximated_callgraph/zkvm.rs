//! Pythran approximated_callgraph: the `size x size` table of `i + j`, built
//! row by row through a mapped lambda.
//!
//! ```python
//! def call(i, j):
//!     return i + j
//!
//! @requires(lambda size: 0 <= size <= 1024)
//! def approximated_callgraph(size: Integer):
//!     out = list()
//!     for i in range(size):
//!         out.append(list(map(lambda j: call(i, j), range(size))))
//!     return out
//! ```
use crate::prelude::*;

pub mod approximated_callgraph {
    use super::*;

    /// The list of rows, as a matrix.
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let size: i64 = args.scalar();
        assert!((0..=1024).contains(&size), "requires 0 <= size <= 1024");
        let n = size as usize;
        Array2::from_shape_fn((n, n), |(i, j)| (i + j) as i64)
    }
}
