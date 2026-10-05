//! Pythran guerre: Laguerre's method for a root of the complex polynomial
//! `a[0] + a[1] z + ... + a[n] z^n`, from the starting point `z`.
//!
//! ```python
//! @requires(lambda a, n, z, err, nter: 0 <= n <= 1024)
//! def guerre(a: Complex[Array, '4'], n: Integer, z: Complex, err: Float, nter: Integer):
//!     az = [complex(0, 0) for i in range(50)]
//!     azz = [complex(0, 0) for i in range(50)]
//!     ci = complex(0.0, 1.0)
//!     eps = 1e-20
//!     for i in range(1, n + 1):
//!         az[i - 1] = float(i) * a[i]
//!     for i in range(1, n):
//!         azz[i - 1] = float(i) * az[i]
//!     dz = err + 1
//!     itera = 0
//!     jter = 0
//!     while abs(dz) > err and itera < nter:
//!         p = a[n - 1] + a[n] * z
//!         for i in range(n - 1, 0, -1):
//!             p = a[i - 1] + z * p
//!         if abs(p) < eps:
//!             return z
//!         pz = az[n - 2] + az[n - 1] * z
//!         for i in range(n - 2, 0, -1):
//!             pz = az[i - 1] + z * pz
//!         pzz = azz[n - 3] + azz[n - 2] * z
//!         for i in range(n - 3, 0, -1):
//!             pzz = azz[i - 1] + z * pzz
//!         f = pz / p
//!         g = f ** 2 - pzz / p
//!         h = n * g
//!         amp1 = abs(f + h)
//!         amp2 = abs(f - h)
//!         if amp1 > amp2:
//!             dz = float(-n) / (f + h)
//!         else:
//!             dz = float(-n) / (f - h)
//!         itera = itera + 1
//!         jter = jter + 1
//!         if jter == 10:
//!             jter = 1
//!             dz = dz * ci
//!         z = z + dz
//!         if jter == 100:
//!             raise RuntimeError('Laguerre method not converging')
//!     return z
//! ```
//!
//! `jter` is reset to 1 when it reaches 10, so the `jter == 100` raise is
//! unreachable and left out.
use crate::prelude::*;

/// Length of the derivative coefficient lists `az` and `azz`.
const LEN: usize = 50;
/// `round(1e-20 * 2^32) = 0`: `abs(p) < eps` never holds in Q32.
const EPS: Q32 = Q32::from_bits(0);

/// Horner's rule as the kernel writes it: `c[m - 1] + c[m] * z`, then
/// `c[i - 1] + z * p` for `i = m - 1, ..., 1`. List indices wrap like
/// Python's (`m` may be below 2).
fn horner(c: &[C32], m: i64, z: C32) -> C32 {
    let at = |i: i64| c[np_index(i, c.len())];
    let mut p = at(m - 1) + at(m) * z;
    for i in (1..m).rev() {
        p = at(i - 1) + z * p;
    }
    p
}

pub mod guerre {
    use super::*;

    pub type Output = C32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let a: Array1<C32> = args.array1();
        let n: i64 = args.scalar();
        let mut z: C32 = args.scalar();
        let err: Q32 = args.scalar();
        let nter: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= n <= 1024");
        let a = a.as_slice().expect("standard layout");

        // `float(i) * c` scales both parts by an integer, exactly.
        let mut az = [C32::ZERO; LEN];
        for i in 1..=n as usize {
            az[i - 1] = a[i] * Q32::from_int(i as i64);
        }
        let mut azz = [C32::ZERO; LEN];
        for i in 1..n as usize {
            azz[i - 1] = az[i] * Q32::from_int(i as i64);
        }

        let mut dz = C32::from_real(err + Q32::ONE);
        let mut itera = 0;
        let mut jter = 0;
        while dz.abs() > err && itera < nter {
            let p = horner(a, n, z);
            if p.abs() < EPS {
                return z;
            }
            let pz = horner(&az, n - 1, z);
            let pzz = horner(&azz, n - 2, z);
            let f = pz / p;
            // `f ** 2` is `f * f` (NumPy squares a complex scalar by one product).
            let g = f * f - pzz / p;
            let h = g * Q32::from_int(n);
            let (amp1, amp2) = ((f + h).abs(), (f - h).abs());
            let minus_n = C32::from_real(Q32::from_int(-n));
            dz = minus_n / if amp1 > amp2 { f + h } else { f - h };
            itera += 1;
            jter += 1;
            if jter == 10 {
                jter = 1;
                dz = dz * C32::I;
            }
            z = z + dz;
        }
        z
    }
}
