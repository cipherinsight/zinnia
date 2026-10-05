//! Pythran spectraldns: two steps of a spectral Navier-Stokes solver on
//! fields with three components (the leading axis).
//!
//! ```python
//! def cross1(c, a, b):
//!     c[0] = a[0] * b[2] - a[2] * b[1]
//!     c[1] = a[2] * b[0] - a[0] * b[2]
//!     c[2] = a[0] * b[1] - a[1] * b[0]
//!     return c
//!
//! def cross2(c: Complex[Array, '3 2 5 7'], a: Float[Array, '3 2 5 7'], b: Complex[Array, '3 2 5 7']):
//!     cross1(c, a, b)
//!     c *= 1j
//!     return c
//!
//! def add_pressure_diffusion(dU: Complex[Array, '3 2 5 7'], U_hat: Complex[Array, '3 2 5 7'], K2: Float[Array, '2 5 7'], K: Float[Array, '3 2 5 7'], P_hat: Complex[Array, '2 5 7'], K_over_K2: Float[Array, '3 2 5 7'], nu: Float):
//!     du0, du1, du2 = (dU[0], dU[1], dU[2])
//!     k_0, k_1, k_2 = (K_over_K2[0], K_over_K2[1], K_over_K2[2])
//!     P_hat[:] = du0 * k_0 + du1 * k_1 + du2 * k_2
//!     dU[:] = dU - P_hat * K - nu * K2 * U_hat
//!     return P_hat
//! ```
use crate::prelude::*;

pub mod cross2 {
    use super::*;

    pub type Output = Array4<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut c: Array4<C32> = args.array4();
        let a: Array4<Q32> = args.array4();
        let b: Array4<C32> = args.array4();
        // cross1: component `out` of c is `a[i] * b[j] - a[k] * b[l]` (upstream's
        // c[0] is not the usual first component of a cross product; kept as
        // written). A real times a complex number scales both parts.
        for (out, (i, j), (k, l)) in [
            (0, (0, 2), (2, 1)),
            (1, (2, 0), (0, 2)),
            (2, (0, 1), (1, 0)),
        ] {
            Zip::from(c.index_axis_mut(Axis(0), out))
                .and(a.index_axis(Axis(0), i))
                .and(b.index_axis(Axis(0), j))
                .and(a.index_axis(Axis(0), k))
                .and(b.index_axis(Axis(0), l))
                .for_each(|c, &ai, &bj, &ak, &bl| *c = bj * ai - bl * ak);
        }
        // `c *= 1j`: (re + im j) * 1j = -im + re j, exactly.
        c.mapv_inplace(|z| C32::new(-z.im, z.re));
        c
    }
}

pub mod add_pressure_diffusion {
    use super::*;

    // The update of dU is not an output (the function returns P_hat), so the
    // parameters only it reads are unused; Zinnia's dead-code elimination
    // drops them as well.
    // unused parameters: U_hat, K2, K, nu
    pub type Output = Array3<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let du: Array4<C32> = args.array4();
        let mut p_hat: Array3<C32> = args.array3();
        let k_over_k2: Array4<Q32> = args.array4();
        assert_eq!(
            &du.shape()[1..],
            p_hat.shape(),
            "could not broadcast into P_hat"
        );
        // P_hat[:] = du0 * k_0 + du1 * k_1 + du2 * k_2
        Zip::indexed(&mut p_hat).for_each(|(x, y, z), p| {
            let term = |c: usize| du[(c, x, y, z)] * k_over_k2[(c, x, y, z)];
            *p = term(0) + term(1) + term(2);
        });
        p_hat
    }
}
