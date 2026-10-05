//! Pythran pselect: picks one of two functions by the truth of `n` and of
//! `not n`; each appends one value to a list.
//!
//! ```python
//! @requires(lambda n: 0 <= n <= 1024)
//! def pselect(n: Integer):
//!     l = list()
//!     for k in (n, not n):
//!         if k:
//!             a = sel0
//!         else:
//!             a = sel1
//!         a(l)
//!     return l
//!
//! def sel0(n):
//!     n.append(1)
//!
//! def sel1(n):
//!     n.append(2.0)
//! ```
//!
//! The list holds an int and a float (`[1, 2.0]` or `[2.0, 1]`); as a NumPy
//! array it is float64, so both are output as floats.
use crate::prelude::*;

pub mod pselect {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let select = |k: bool| if k { Q32::ONE } else { Q32::from_int(2) };
        Array1::from(vec![select(n != 0), select(n == 0)])
    }
}
