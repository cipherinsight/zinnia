//! Pythran repeating: `x` followed by every element of `x` repeated
//! `nvar_y` times.
//!
//! ```python
//! def repeating(x: Float[Array, '10'], nvar_y: Integer):
//!     nvar_x = x.shape[0]
//!     y = np.empty(nvar_x * (1 + nvar_y))
//!     y[0:nvar_x] = x[0:nvar_x]
//!     y[nvar_x:] = np.repeat(x, nvar_y)
//!     return y
//! ```
use crate::prelude::*;

pub mod repeating {
    use super::*;

    /// `nvar_x * (1 + nvar_y)` values.
    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<Q32> = args.array1();
        let nvar_y: i64 = args.scalar();
        let repeats = usize::try_from(nvar_y).expect("repeats may not contain negative values");
        let mut y = Vec::with_capacity(x.len() * (1 + repeats));
        y.extend(x.iter().copied());
        for &v in &x {
            y.extend(core::iter::repeat_n(v, repeats));
        }
        Array1::from(y)
    }
}
