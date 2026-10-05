//! Q32.32 fixed-point numbers for the SP1 and RISC Zero baselines.
//!
//! A real value `v` is stored as the integer `round(v * 2^32)` in an `i64`,
//! exactly like Zinnia's quantizer (`src/prove/field.rs::quantize`). The
//! arithmetic follows Zinnia's fixed-point semantics (`src/prove/kernel.rs`):
//!
//! * `a * b` is `floor(a * b / 2^32)` (Zinnia `fp_mul_rescale`), computed in
//!   `i128` and rescaled with an arithmetic shift, which rounds toward
//!   negative infinity.
//! * `a / b` is `floor(a * 2^32 / b)` with Euclidean rounding (Zinnia
//!   `fp_div_prescale`).
//! * [`dot`] accumulates the raw products and rescales once (Zinnia
//!   `fp_product_sum_rescale`, halo2-graph `inner_product`).
//! * [`Q32::sqrt`] is `floor(sqrt(a * 2^32))`, the exact integer square root
//!   (Zinnia `fp_sqrt`).
//! * [`Q32::sin`], [`Q32::cos`] and [`Q32::asin`] are Zinnia's fixed-point
//!   algorithms (`fp_sin`, `fp_cos`, `fp_arcsin`, which the halo2-base circuit
//!   constrains): range reduction to `[0, pi/4]` and Taylor polynomials in
//!   `s^2` for sin/cos, `2 * atan(x / (1 + sqrt(1 - x^2)))` with an odd atan
//!   polynomial for arcsin; Horner's rule rescales every product. Error
//!   against libm: about 1e-9 for sin/cos on `[-20, 20]`, 6e-9 for arcsin.
//!
//! The crate is `no_std` and never touches floating point unless the `std`
//! feature is enabled; that feature only adds the host-side encoding and
//! decoding of `f64` values.
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use serde::{Deserialize, Serialize};

/// Number of fractional bits (Zinnia `precision_bits`).
pub const FRAC_BITS: u32 = 32;
const SCALE: i64 = 1 << FRAC_BITS;

/// A Q32.32 fixed-point number (see the crate documentation).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Q32(i64);

impl Q32 {
    pub const ZERO: Q32 = Q32(0);
    pub const ONE: Q32 = Q32(SCALE);
    /// `round(pi * 2^32)`.
    pub const PI: Q32 = Q32(13_493_037_705);
    /// `round(pi / 2 * 2^32)`.
    pub const FRAC_PI_2: Q32 = Q32(6_746_518_852);
    /// `round(pi / 4 * 2^32)`.
    pub const FRAC_PI_4: Q32 = Q32(3_373_259_426);
    /// `round(pi / 180 * 2^32)`, the factor of `np.deg2rad`.
    pub const DEG_TO_RAD: Q32 = Q32(74_961_321);
    /// `round(180 / pi * 2^32)`, the factor of `np.degrees`.
    pub const RAD_TO_DEG: Q32 = Q32(246_083_499_208);

    /// The value whose fixed-point encoding is `bits`.
    pub const fn from_bits(bits: i64) -> Q32 {
        Q32(bits)
    }

    /// The fixed-point encoding of `self`.
    pub const fn to_bits(self) -> i64 {
        self.0
    }

    /// The exact fixed-point value of an integer (Zinnia `float_cast`).
    pub const fn from_int(n: i64) -> Q32 {
        Q32(n << FRAC_BITS)
    }

    /// Python `int(x)` / NumPy `astype(int)`: truncate toward zero.
    pub const fn trunc_to_int(self) -> i64 {
        self.0 / SCALE
    }

    /// `np.floor`.
    pub const fn floor(self) -> Q32 {
        Q32(self.0 & !(SCALE - 1))
    }

    pub const fn abs(self) -> Q32 {
        Q32(self.0.abs())
    }

    pub const fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// `np.sqrt` for a non-negative argument: `floor(sqrt(x * 2^32))`.
    pub fn sqrt(self) -> Q32 {
        assert!(self.0 >= 0, "sqrt of a negative number");
        let root = ((self.0 as u128) << FRAC_BITS).isqrt();
        Q32(root as i64)
    }

    /// `np.sin` (Zinnia `fp_sin`): reduce `|x|` modulo `2*pi`, fold onto
    /// `[0, pi)` (`sin(r) = -sin(r - pi)`), mirror onto `[0, pi/2]`
    /// (`sin(r) = sin(pi - r)`), and above `pi/4` use `cos(pi/2 - r)`, so the
    /// Taylor polynomials run on `s` in `[0, pi/4]`, in `u = s^2`.
    pub fn sin(self) -> Q32 {
        let magnitude = self.0.abs();
        let r = magnitude % (2 * Self::PI.0);
        let (r, folded) = if r >= Self::PI.0 { (r - Self::PI.0, true) } else { (r, false) };
        let r = if r > Self::FRAC_PI_2.0 { Self::PI.0 - r } else { r };
        let value = if r > Self::FRAC_PI_4.0 {
            let s = Q32(Self::FRAC_PI_2.0 - r);
            horner(s * s, &COS_TAYLOR)
        } else {
            let s = Q32(r);
            s * horner(s * s, &SIN_OVER_X_TAYLOR)
        };
        if self.is_negative() != folded {
            -value
        } else {
            value
        }
    }

    /// `np.cos(x) = sin(x + pi/2)` (Zinnia `fp_cos`).
    pub fn cos(self) -> Q32 {
        (self + Self::FRAC_PI_2).sin()
    }

    /// `np.arcsin` for `x` in `[-1, 1]` (Zinnia `fp_arcsin`):
    /// `2 * atan(x / (1 + sqrt(1 - x^2)))`. The root is the exact integer root
    /// of `2^64 - x^2` in raw units, and `atan(t) = t * P(t^2)` with `t` in
    /// `[-1, 1]`.
    pub fn asin(self) -> Q32 {
        assert!(-Self::ONE <= self && self <= Self::ONE, "arcsin argument outside [-1, 1]");
        let one = Self::ONE.0 as i128;
        let x = self.0 as i128;
        let root = ((one * one - x * x) as u128).isqrt() as i64;
        let t = self / Q32(Self::ONE.0 + root);
        let half = t * horner(t * t, &ATAN_OVER_X);
        half + half
    }
}

/// Horner evaluation with the coefficients given from the highest degree
/// down; every step rescales the product (Zinnia `horner_eval`).
fn horner(x: Q32, coefs_high_first: &[Q32]) -> Q32 {
    let (&first, rest) = coefs_high_first.split_first().expect("a polynomial has coefficients");
    rest.iter().fold(first, |acc, &c| acc * x + c)
}

/// `round(c * 2^32)` of `(-1)^k / (2k+1)!` for k = 6 down to 0: Taylor
/// coefficients of `sin(s) / s` in `u = s^2` (Zinnia `sin_taylor_q`).
const SIN_OVER_X_TAYLOR: [Q32; 7] = [
    Q32(1),
    Q32(-108),
    Q32(11_836),
    Q32(-852_176),
    Q32(35_791_394),
    Q32(-715_827_883),
    Q32(4_294_967_296),
];

/// `round(c * 2^32)` of `(-1)^k / (2k)!` for k = 7 down to 0: Taylor
/// coefficients of `cos(s)` in `u = s^2` (Zinnia `cos_taylor_q`).
const COS_TAYLOR: [Q32; 8] = [
    Q32(0),
    Q32(9),
    Q32(-1_184),
    Q32(106_522),
    Q32(-5_965_232),
    Q32(178_956_971),
    Q32(-2_147_483_648),
    Q32(4_294_967_296),
];

/// `round(c * 2^32)` of Zinnia's `ATAN_ODD_U_COEFS` (`atan(t) / t` as a
/// degree-9 polynomial in `u = t^2`, a least-squares fit on `[-1, 1]`),
/// highest degree first.
const ATAN_OVER_X: [Q32; 10] = [
    Q32(-7_489_138),
    Q32(45_873_335),
    Q32(-131_930_687),
    Q32(246_804_093),
    Q32(-359_577_404),
    Q32(469_877_966),
    Q32(-612_529_925),
    Q32(858_917_466),
    Q32(-1_431_653_573),
    Q32(4_294_967_285),
];

/// Fused inner product `floor(sum(a_i * b_i) / 2^32)`: the raw products are
/// accumulated in `i128` and rescaled once.
pub fn dot(pairs: impl IntoIterator<Item = (Q32, Q32)>) -> Q32 {
    let raw: i128 = pairs.into_iter().map(|(a, b)| a.0 as i128 * b.0 as i128).sum();
    Q32((raw >> FRAC_BITS) as i64)
}

impl Add for Q32 {
    type Output = Q32;
    fn add(self, rhs: Q32) -> Q32 {
        Q32(self.0 + rhs.0)
    }
}

impl Sub for Q32 {
    type Output = Q32;
    fn sub(self, rhs: Q32) -> Q32 {
        Q32(self.0 - rhs.0)
    }
}

impl Neg for Q32 {
    type Output = Q32;
    fn neg(self) -> Q32 {
        Q32(-self.0)
    }
}

/// `floor(a * b / 2^32)`.
impl Mul for Q32 {
    type Output = Q32;
    fn mul(self, rhs: Q32) -> Q32 {
        Q32(((self.0 as i128 * rhs.0 as i128) >> FRAC_BITS) as i64)
    }
}

/// Multiplication by an integer is exact.
impl Mul<i64> for Q32 {
    type Output = Q32;
    fn mul(self, rhs: i64) -> Q32 {
        Q32(self.0 * rhs)
    }
}

/// `floor(a * 2^32 / b)` with Euclidean rounding; panics if `b == 0`.
impl Div for Q32 {
    type Output = Q32;
    fn div(self, rhs: Q32) -> Q32 {
        Q32(((self.0 as i128) << FRAC_BITS).div_euclid(rhs.0 as i128) as i64)
    }
}

/// Division by an integer `n`, equal to `self / Q32::from_int(n)`.
impl Div<i64> for Q32 {
    type Output = Q32;
    fn div(self, rhs: i64) -> Q32 {
        Q32(self.0.div_euclid(rhs))
    }
}

impl AddAssign for Q32 {
    fn add_assign(&mut self, rhs: Q32) {
        *self = *self + rhs;
    }
}

impl SubAssign for Q32 {
    fn sub_assign(&mut self, rhs: Q32) {
        *self = *self - rhs;
    }
}

impl MulAssign for Q32 {
    fn mul_assign(&mut self, rhs: Q32) {
        *self = *self * rhs;
    }
}

impl MulAssign<i64> for Q32 {
    fn mul_assign(&mut self, rhs: i64) {
        *self = *self * rhs;
    }
}

impl DivAssign for Q32 {
    fn div_assign(&mut self, rhs: Q32) {
        *self = *self / rhs;
    }
}

impl DivAssign<i64> for Q32 {
    fn div_assign(&mut self, rhs: i64) {
        *self = *self / rhs;
    }
}

impl core::iter::Sum for Q32 {
    fn sum<I: Iterator<Item = Q32>>(iter: I) -> Q32 {
        iter.fold(Q32::ZERO, Add::add)
    }
}

impl<'a> core::iter::Sum<&'a Q32> for Q32 {
    fn sum<I: Iterator<Item = &'a Q32>>(iter: I) -> Q32 {
        iter.copied().sum()
    }
}

/// Python integer division and remainder (round toward negative infinity),
/// which NumPy uses for `//` and `%` on integers.
pub mod int {
    pub const fn floor_div(a: i64, b: i64) -> i64 {
        let q = a / b;
        if (a % b != 0) && ((a < 0) != (b < 0)) {
            q - 1
        } else {
            q
        }
    }

    pub const fn floor_mod(a: i64, b: i64) -> i64 {
        a - b * floor_div(a, b)
    }
}

/// Host-side conversion between `f64` and the fixed-point encoding.
#[cfg(feature = "std")]
impl Q32 {
    /// `round(v * 2^32)`, rounding half away from zero (Zinnia `quantize`).
    pub fn from_f64(v: f64) -> Q32 {
        Q32((v * SCALE as f64).round() as i64)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE as f64
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn close(a: Q32, b: f64, tol: f64) -> bool {
        (a.to_f64() - b).abs() <= tol
    }

    #[test]
    fn rescaling_rounds_toward_negative_infinity() {
        let half = Q32::from_f64(0.5);
        let eps = Q32::from_bits(1);
        assert_eq!(half * eps, Q32::ZERO);
        assert_eq!(-half * eps, Q32::from_bits(-1));
        assert_eq!(Q32::ONE / Q32::from_int(3), Q32::from_bits(1_431_655_765));
        assert_eq!(-Q32::ONE / Q32::from_int(3), Q32::from_bits(-1_431_655_766));
        assert_eq!(Q32::from_int(-7) / 2, Q32::from_f64(-3.5));
    }

    #[test]
    fn transcendentals_match_libm() {
        // Against libm on the encoded argument, on the grids Zinnia's kernel
        // documents (max error 1.1e-9 for sin/cos, 5.9e-9 for arcsin).
        for i in -4000..=4000 {
            let q = Q32::from_f64(i as f64 / 200.0);
            assert!(close(q.sin(), q.to_f64().sin(), 2e-9), "sin {q:?}");
            assert!(close(q.cos(), q.to_f64().cos(), 2e-9), "cos {q:?}");
        }
        for i in -4000..=4000 {
            let q = Q32::from_f64(i as f64 / 4000.0);
            assert!(close(q.asin(), q.to_f64().asin(), 1e-8), "asin {q:?}");
        }
        for v in [0.0, 1e-9, 0.25, 2.0, 10.5, 12345.678] {
            // Exact up to the final floor: compare with the root of the encoded input.
            let q = Q32::from_f64(v);
            assert!(close(q.sqrt(), q.to_f64().sqrt(), 1.0 / SCALE as f64), "sqrt {v}");
        }
    }

    #[test]
    fn python_integer_division() {
        assert_eq!(int::floor_div(-7, 2), -4);
        assert_eq!(int::floor_mod(-7, 2), 1);
        assert_eq!(int::floor_div(7, -2), -4);
        assert_eq!(int::floor_mod(7, -2), -1);
        assert_eq!(int::floor_div(6, 3), 2);
    }
}
