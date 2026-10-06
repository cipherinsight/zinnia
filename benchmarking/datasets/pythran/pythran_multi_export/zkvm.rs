//! Pythran multi_export: the identity on a float.
//!
//! ```python
//! def a(i: Float):
//!     return i
//! ```
use crate::prelude::*;

pub mod a {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        args.scalar()
    }
}
