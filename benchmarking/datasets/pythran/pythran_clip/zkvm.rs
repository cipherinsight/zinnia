//! Pythran clip: scales every complex entry whose magnitude exceeds `_max`
//! down to magnitude `_max`.
//!
//! ```python
//! def limit1(x, epsilon=1e-06):
//!     if abs(x) < epsilon:
//!         return 0
//!     else:
//!         return x / abs(x)
//!
//! def clip(z: Complex[Array, '2'], _max: Float):
//!     out = np.empty(z.shape, dtype=z.dtype)
//!     for i in range(len(z)):
//!         if abs(z[i]) > _max:
//!             out[i] = limit1(z[i]) * _max
//!         else:
//!             out[i] = z[i]
//!     return out
//! ```
use crate::prelude::*;

/// `round(1e-6 * 2^32)`.
const EPSILON: Q32 = Q32::from_bits(4295);

/// `x / abs(x)`, or 0 when `abs(x) < 1e-6`.
fn limit1(x: C32) -> C32 {
    let magnitude = x.abs();
    if magnitude < EPSILON {
        C32::ZERO
    } else {
        x / magnitude
    }
}

pub mod clip {
    use super::*;

    pub type Output = Array1<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let z: Array1<C32> = args.array1();
        let max: Q32 = args.scalar();
        z.mapv(|zi| if zi.abs() > max { limit1(zi) * max } else { zi })
    }
}
