//! Pythran deriv: successive derivatives `dg` of a complex function from the
//! derivatives of its three terms `dh1`, `dh2`, `dh3` and binomial
//! coefficients `bin` (Faa di Bruno style recurrence).
//!
//! ```python
//! @requires(lambda n, sig, alp, dg, dh1, dh3, bin, nu: 0 <= n <= 1024)
//! def deriv(n: Integer, sig: Float, alp: Complex, dg: Complex[Array, '4'], dh1: Complex[Array, '4'], dh3: Complex[Array, '4'], bin: Complex[Array, '5 4'], nu: Float):
//!     dh2 = [complex(0, 0) for _ in dh1]
//!     ci = complex(0.0, 1.0)
//!     dh1[0] = complex(0.5, 0) * ci * complex(sig, 0)
//!     exp1 = complex(-0.5, 0)
//!     dh2[0] = alp
//!     exp2 = complex(-1.0, 0)
//!     dh3[0] = -2.0 * nu
//!     exp3 = complex(-1.0, 0)
//!     for i in range(1, n):
//!         dh1[i] = dh1[i - 1] * exp1
//!         exp1 = exp1 - 1.0
//!         dh2[i] = dh2[i - 1] * exp2
//!         exp2 = exp2 - 1.0
//!         dh3[i] = -nu * dh3[i - 1] * exp3
//!         exp3 = exp3 - 1.0
//!     dg[0] = complex(1.0)
//!     dg[1] = dh1[0] + dh2[0] + dh3[0]
//!     for i in range(2, n + 1):
//!         dg[i] = dh1[i - 1] + dh2[i - 1] + dh3[i - 1]
//!         for j in range(1, i):
//!             dg[i] = dg[i] + bin[j - 1][i - 1] * (dh1[j - 1] + dh2[j - 1] + dh3[j - 1]) * dg[i - j]
//!     return dg
//! ```
use crate::prelude::*;

pub mod deriv {
    use super::*;

    pub type Output = Array1<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let sig: Q32 = args.scalar();
        let alp: C32 = args.scalar();
        let mut dg: Array1<C32> = args.array1();
        let mut dh1: Array1<C32> = args.array1();
        let mut dh3: Array1<C32> = args.array1();
        let bin: Array2<C32> = args.array2();
        let nu: Q32 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let n = n as usize;

        let mut dh2 = Array1::<C32>::zeros(dh1.len());
        dh1[0] = C32::from_real(Q32::HALF) * C32::I * C32::from_real(sig);
        dh2[0] = alp;
        dh3[0] = C32::from_real(nu * -2);
        // `exp1`, `exp2`, `exp3`: complex numbers whose real part steps down by 1.
        let (mut exp1, mut exp2, mut exp3) = (C32::from_real(-Q32::HALF), -C32::ONE, -C32::ONE);
        for i in 1..n {
            dh1[i] = dh1[i - 1] * exp1;
            exp1 = exp1 - C32::ONE;
            dh2[i] = dh2[i - 1] * exp2;
            exp2 = exp2 - C32::ONE;
            // `-nu * dh3[i - 1] * exp3`: a real times a complex number, then
            // a complex product.
            dh3[i] = dh3[i - 1] * -nu * exp3;
            exp3 = exp3 - C32::ONE;
        }

        let dh = |k: usize| dh1[k] + dh2[k] + dh3[k];
        dg[0] = C32::ONE;
        dg[1] = dh(0);
        for i in 2..=n {
            dg[i] = dh(i - 1);
            for j in 1..i {
                dg[i] = dg[i] + bin[(j - 1, i - 1)] * dh(j - 1) * dg[i - j];
            }
        }
        dg
    }
}
