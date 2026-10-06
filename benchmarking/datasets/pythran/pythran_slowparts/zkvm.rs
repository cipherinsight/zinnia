//! Pythran slowparts: the linear-algebra intensive part of the gradients of
//! an autoencoder model (the "grae" of the kernel's docstring).
//!
//! ```python
//! from numpy import zeros, power, tanh
//!
//! def slowparts(d: Integer, re: Integer, preDz: Float[Array, '20 11 5'], preWz: Float[Array, '10 11 5'], SRW: Float[Array, '10 20'], RSW: Float[Array, '10 20'], yxV: Float[Array, '11 5 10'], xyU: Float[Array, '11 5 10'], resid: Integer):
//!     fprime = lambda x: 1 - power(tanh(x), 2)
//!     partialDU = zeros((d + 1, re, 2 * d, d))
//!     for k in range(2 * d):
//!         for i in range(d):
//!             partialDU[:, :, k, i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i, k] + RSW[i, k]) * yxV[:, :, i]
//!     return partialDU
//! ```
//!
//! `fprime(preDz[k])` and `fprime(preWz[i])` do not depend on the other loop
//! variable, so they are computed once per slice.
use crate::prelude::*;

/// `fprime(x) = 1 - tanh(x) ** 2`.
fn fprime(x: Q32) -> Q32 {
    Q32::ONE - x.tanh().powi(2)
}

#[allow(non_snake_case)]
pub mod slowparts {
    use super::*;

    // unused parameters: xyU, resid
    pub type Output = Array4<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let d: i64 = args.scalar();
        let re: i64 = args.scalar();
        let pre_dz: Array3<Q32> = args.array3();
        let pre_wz: Array3<Q32> = args.array3();
        let srw: Array2<Q32> = args.array2();
        let rsw: Array2<Q32> = args.array2();
        let yxv: Array3<Q32> = args.array3();
        let dim = |n: i64| usize::try_from(n).expect("negative dimensions are not allowed");
        let (d, re) = (dim(d), dim(re));

        let fprime_dz = pre_dz.mapv(fprime);
        let fprime_wz = pre_wz.mapv(fprime);
        let mut partial_du = Array4::<Q32>::zeros((d + 1, re, 2 * d, d));
        for k in 0..2 * d {
            for i in 0..d {
                let weight = srw[(i, k)] + rsw[(i, k)];
                let term = &fprime_dz.index_axis(Axis(0), k) * &fprime_wz.index_axis(Axis(0), i);
                let term = term.mapv(|v| v * weight) * yxv.index_axis(Axis(2), i);
                partial_du.slice_mut(s![.., .., k, i]).assign(&term);
            }
        }
        partial_du
    }
}
