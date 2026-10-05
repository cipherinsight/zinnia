//! Pythran babylonian: whether an integer is a perfect square, by the
//! Babylonian (Newton) iteration on integers, stopping when it cycles.
//!
//! ```python
//! def is_square(a_positive_int: Integer):
//!     """ Check if it is a perfect square. """
//!     x = a_positive_int // 2
//!     seen = {x}
//!     while x * x != a_positive_int:
//!         x = (x + a_positive_int // x) // 2
//!         if x in seen:
//!             return False
//!         seen.add(x)
//!     return True
//! ```
//!
//! Python integers do not overflow, so `x * x` is compared in `i128`. As in
//! Python, `a_positive_int == 1` divides by zero (`x = 0`) and panics.
use alloc::collections::BTreeSet;

use crate::prelude::*;

pub mod is_square {
    use super::*;

    pub type Output = bool;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: i64 = args.scalar();
        let mut x = floor_div(a, 2);
        let mut seen = BTreeSet::from([x]);
        while i128::from(x) * i128::from(x) != i128::from(a) {
            x = floor_div(x + floor_div(a, x), 2);
            if !seen.insert(x) {
                return false;
            }
        }
        true
    }
}
