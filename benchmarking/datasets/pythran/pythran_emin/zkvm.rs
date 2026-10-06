//! Pythran emin: energy minimisation of a water molecule's internal
//! coordinates by a fixed number of gradient-descent steps.
//!
//! ```python
//! def Eandg(rOH, thetaHOH):
//!     kOH = 50.0
//!     rOHe = 0.95
//!     kHOH = 50.0
//!     thetaHOHe = 104.5
//!     E = 2 * kOH * (rOH - rOHe) ** 2 + kHOH * (thetaHOH - thetaHOHe) ** 2
//!     grOH = 2 * kOH * (rOH - rOHe)
//!     grthetaHOH = 2 * kHOH * (thetaHOH - thetaHOHe)
//!     return (E, grOH, grthetaHOH)
//!
//! def run():
//!     c = 0.005
//!     n_steps = 1000000
//!     rOH = 10.0
//!     thetaHOH = 180.0
//!     for i in range(n_steps):
//!         E, grOH, gthetaHOH = Eandg(rOH, thetaHOH)
//!         if abs(grOH) > 0.001 / c or abs(gthetaHOH) > 0.01 / c:
//!             rOH = rOH - c * grOH
//!             thetaHOH = thetaHOH - c * gthetaHOH
//!     converged = abs(grOH) > 0.001 / c or abs(gthetaHOH) > 0.01 / c
//!     return (converged, E, rOH, thetaHOH)
//! ```
use crate::prelude::*;

/// `kOH = kHOH = 50.0`.
const K: i64 = 50;
/// `rOHe = 0.95`: `round(0.95 * 2^32)`.
const R_OH_E: Q32 = Q32::from_bits(4_080_218_931);
/// `thetaHOHe = 104.5`, exact in Q32.
const THETA_HOH_E: Q32 = Q32::from_bits(209 << 31);
/// `c = 0.005`: `round(0.005 * 2^32)`.
const C: Q32 = Q32::from_bits(21_474_836);
/// `0.001`: `round(0.001 * 2^32)`.
const R_TOL: Q32 = Q32::from_bits(4_294_967);
/// `0.01`: `round(0.01 * 2^32)`.
const THETA_TOL: Q32 = Q32::from_bits(42_949_673);
/// `n_steps`.
const N_STEPS: usize = 1_000_000;

/// `Eandg`: the force-field energy and its two gradients.
fn energy_and_gradient(r_oh: Q32, theta_hoh: Q32) -> (Q32, Q32, Q32) {
    let (dr, dtheta) = (r_oh - R_OH_E, theta_hoh - THETA_HOH_E);
    // `2 * kOH * (rOH - rOHe) ** 2` is `(2 * kOH) * (dr ** 2)`; `2 * kOH` is
    // the integer-valued 100.0, so those products are exact.
    let energy = (dr * dr) * (2 * K) + (dtheta * dtheta) * K;
    (energy, dr * (2 * K), dtheta * (2 * K))
}

pub mod run {
    use super::*;

    pub type Output = (bool, Q32, Q32, Q32);

    pub fn main(_args: &mut Args<impl Input>) -> Output {
        // `0.001 / c` and `0.01 / c`, evaluated once.
        let (r_bound, theta_bound) = (R_TOL / C, THETA_TOL / C);
        let not_converged = |gr: Q32, gtheta: Q32| gr.abs() > r_bound || gtheta.abs() > theta_bound;
        let mut r_oh = Q32::from_int(10);
        let mut theta_hoh = Q32::from_int(180);
        let (mut energy, mut gr, mut gtheta) = (Q32::ZERO, Q32::ZERO, Q32::ZERO);
        for _ in 0..N_STEPS {
            (energy, gr, gtheta) = energy_and_gradient(r_oh, theta_hoh);
            if not_converged(gr, gtheta) {
                r_oh = r_oh - C * gr;
                theta_hoh = theta_hoh - C * gtheta;
            }
        }
        (not_converged(gr, gtheta), energy, r_oh, theta_hoh)
    }
}
