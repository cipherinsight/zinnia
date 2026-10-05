//! Pythran hasting: the right-hand side of the Hastings-Powell three-species
//! food-web ODE.
//!
//! ```python
//! def fweb(y: Float[Array, '8'], t: Float, a1: Float, a2: Float, b1: Float, b2: Float, d1: Float, d2: Float):
//!     yprime = np.empty((3,))
//!     yprime[0] = y[0] * (1.0 - y[0]) - a1 * y[0] * y[1] / (1.0 + b1 * y[0])
//!     yprime[1] = a1 * y[0] * y[1] / (1.0 + b1 * y[0]) - a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d1 * y[1]
//!     yprime[2] = a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d2 * y[2]
//!     return yprime
//! ```
use crate::prelude::*;

pub mod fweb {
    use super::*;

    // unused parameters: t
    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let y: Array1<Q32> = args.array1();
        let [a1, a2, b1, b2, d1, d2]: [Q32; 6] = core::array::from_fn(|_| args.scalar());
        let (y0, y1, y2) = (y[0], y[1], y[2]);
        // The two interaction terms appear twice each; Python evaluates both
        // occurrences to the same value.
        let prey = a1 * y0 * y1 / (Q32::ONE + b1 * y0);
        let predator = a2 * y1 * y2 / (Q32::ONE + b2 * y1);
        Array1::from(vec![
            y0 * (Q32::ONE - y0) - prey,
            prey - predator - d1 * y1,
            predator - d2 * y2,
        ])
    }
}
