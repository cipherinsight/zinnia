//! Pythran clip2: clip complex values to magnitude `_max`, keeping their
//! direction.
//!
//! ```python
//! def limit(x, epsilon=1e-06):
//!     out = np.empty(shape=x.shape, dtype=x.dtype)
//!     mask1 = np.abs(x) < epsilon
//!     out[mask1] = 0
//!     mask2 = np.logical_not(mask1)
//!     out[mask2] = x[mask2] / np.abs(x[mask2])
//!     return out
//!
//! def clip(z: Complex[Array, '2'], _max: Float):
//!     mask = np.abs(z) > _max
//!     z[mask] = limit(z[mask]) * _max
//!     return z
//! ```
use crate::prelude::*;

/// `round(1e-6 * 2^32)`, limit's default `epsilon`.
const EPSILON: Q32 = Q32::from_bits(4_295);

pub mod clip {
    use super::*;

    pub type Output = Array1<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut z: Array1<C32> = args.array1();
        let max: Q32 = args.scalar();
        for v in z.iter_mut() {
            let magnitude = v.abs();
            if magnitude > max {
                // limit(v): 0 below epsilon, else the unit vector v / |v|.
                let unit = if magnitude < EPSILON {
                    C32::ZERO
                } else {
                    *v / magnitude
                };
                *v = unit * max;
            }
        }
        z
    }
}
