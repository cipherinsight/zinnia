//! Pythran crotg: BLAS `CROTG`, the complex Givens rotation `(C, S)` that
//! zeroes `CB` against `CA`.
//!
//! ```python
//! def CROTG(CA: Complex, CB: Complex, C: Float=0, S: Complex=0):
//!     if abs(CA) == 0.0:
//!         C = 0.0
//!         S = complex(1.0, 0.0)
//!         CA = CB
//!     else:
//!         SCALE = abs(CA) + abs(CB)
//!         NORM = SCALE * math.sqrt(abs(CA / SCALE) ** 2 + abs(CB / SCALE) ** 2)
//!         ALPHA = CA / abs(CA)
//!         C = abs(CA) / NORM
//!         S = ALPHA * CB.conjugate() / NORM
//!         CA = ALPHA * NORM
//!     return (CA, CB, C, S)
//! ```
use crate::prelude::*;

#[allow(non_snake_case)]
pub mod CROTG {
    use super::*;

    // unused parameters: C, S
    /// `(CA, CB, C, S)`; the kernel overwrites `C` and `S` before reading them.
    pub type Output = (C32, C32, Q32, C32);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let ca: C32 = args.scalar();
        let cb: C32 = args.scalar();
        // `abs(CA) == 0.0`, as written: the magnitude is needed below anyway.
        let abs_ca = ca.abs();
        if abs_ca == Q32::ZERO {
            return (cb, cb, Q32::ZERO, C32::ONE);
        }
        let scale = abs_ca + cb.abs();
        let norm = scale * ((ca / scale).abs().powi(2) + (cb / scale).abs().powi(2)).sqrt();
        let alpha = ca / abs_ca;
        let c = abs_ca / norm;
        let s = alpha * cb.conj() / norm;
        (alpha * norm, cb, c, s)
    }
}
