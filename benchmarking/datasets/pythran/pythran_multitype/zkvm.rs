//! Pythran multitype: three tiny functions exercising Pythran's overloads.
//!
//! ```python
//! @requires(lambda n, m: 0 <= n <= 1024)
//! @requires(lambda n, m: 0 <= m <= 1024)
//! def times(n: Integer, m: Integer):
//!     return n * m
//!
//! def check2(x: Float[Array, '1'], y: Boolean[Array, '1']):
//!     if y:
//!         return (x, y)
//!     else:
//!         return (x + x, y)
//!
//! def check(x: Float[Array, '1'], y: Integer):
//!     if y:
//!         return (x, y)
//!     else:
//!         return (x + x, y)
//! ```
use crate::prelude::*;

pub mod times {
    use super::*;

    pub type Output = i64;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let m: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        assert!((0..=1024).contains(&m), "requires 0 <= m <= 1024");
        n * m
    }
}

pub mod check2 {
    use super::*;

    pub type Output = (Array1<Q32>, Array1<bool>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        let y: Array1<bool> = args.array1();
        // `if y:` takes the truth value of the array, which NumPy defines
        // only for exactly one element.
        assert_eq!(
            y.len(),
            1,
            "the truth value of an array of size != 1 is ambiguous"
        );
        if y[0] {
            (x, y)
        } else {
            (&x + &x, y)
        }
    }
}

pub mod check {
    use super::*;

    pub type Output = (Array1<Q32>, i64);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        let y: i64 = args.scalar();
        if y != 0 {
            (x, y)
        } else {
            (&x + &x, y)
        }
    }
}
