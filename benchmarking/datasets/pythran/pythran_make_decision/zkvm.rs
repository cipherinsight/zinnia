//! Pythran make_decision: hard decision of each received complex sample to
//! the nearest constellation symbol.
//!
//! ```python
//! def md(E: Complex[Array, '160'], symbols: Complex[Array, '80']):
//!     L = E.shape[0]
//!     M = symbols.shape[0]
//!     syms_out = np.zeros(L, dtype=E.dtype)
//!     for i in range(L):
//!         im = np.argmin(abs(E[i] - symbols) ** 2)
//!         syms_out[i] = symbols[im]
//!     return syms_out
//! ```
use crate::prelude::*;

pub mod md {
    use super::*;

    pub type Output = Array1<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let e: Array1<C32> = args.array1();
        let symbols: Array1<C32> = args.array1();
        e.mapv(|ei| {
            // `abs(E[i] - s) ** 2`: the rounded magnitude, squared. `min_by_key`
            // keeps the first minimum, like `np.argmin`.
            let distance = |s: C32| {
                let magnitude = (ei - s).abs();
                magnitude * magnitude
            };
            symbols
                .iter()
                .copied()
                .min_by_key(|&s| distance(s))
                .expect("attempt to get argmin of an empty sequence")
        })
    }
}
