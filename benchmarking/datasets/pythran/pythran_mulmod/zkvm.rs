//! Pythran mulmod: carry-less multiplication of `x` and `y` modulo the
//! polynomial `m` over GF(2) (shift-and-add with conditional reduction).
//!
//! ```python
//! @requires(lambda x, y, m: 0 <= m <= 1024)
//! def gf2mulmod(x: Integer, y: Integer, m: Integer):
//!     z = 0
//!     while x > 0:
//!         if x & 1 != 0:
//!             z ^= y
//!         y <<= 1
//!         y2 = y ^ m
//!         if y2 < y:
//!             y = y2
//!         x >>= 1
//!     return z
//! ```
use crate::prelude::*;

pub mod gf2mulmod {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut x: i64 = args.scalar();
        let mut y: i64 = args.scalar();
        let m: i64 = args.scalar();
        assert!((0..=1024).contains(&m), "requires 0 <= m <= 1024");
        let mut z = 0;
        while x > 0 {
            if x & 1 != 0 {
                z ^= y;
            }
            // Shifting out the top bit wraps like int64.
            y = y.wrapping_shl(1);
            y = y.min(y ^ m);
            x >>= 1;
        }
        z
    }
}
