//! Q32.32 fixed-point numbers for the SP1 and RISC Zero baselines.
//!
//! A real value `v` is stored as the integer `round(v * 2^32)` in an `i64`,
//! exactly like Zinnia's quantizer (`src/prove/field.rs::quantize`). The
//! arithmetic follows Zinnia's fixed-point semantics (`src/prove/kernel.rs`):
//!
//! * `a * b` is `floor(a * b / 2^32)` (Zinnia `fp_mul_rescale`), computed in
//!   `i128` and rescaled with an arithmetic shift, which rounds toward
//!   negative infinity (on 32-bit targets from 32-bit halves instead; see
//!   [`mul_rescaled`]).
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

/// Number of fractional bits (Zinnia `precision_bits`).
pub const FRAC_BITS: u32 = 32;
const SCALE: i64 = 1 << FRAC_BITS;

/// A Q32.32 fixed-point number (see the crate documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Q32(i64);

impl Q32 {
    pub const ZERO: Q32 = Q32(0);
    pub const ONE: Q32 = Q32(SCALE);
    /// `0.5`.
    pub const HALF: Q32 = Q32(SCALE / 2);
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

    /// `np.abs` with Zinnia's domain: `None` for the code `i64::MIN`, whose
    /// magnitude `2^31` has no Q32 code (Zinnia has no proof there; [`Q32::abs`]
    /// wraps it back to `i64::MIN` in release builds).
    pub const fn checked_abs(self) -> Option<Q32> {
        match self.0.checked_abs() {
            Some(bits) => Some(Q32(bits)),
            None => None,
        }
    }

    /// `self - rhs`, or `None` when the difference leaves the Q32 range.
    pub const fn checked_sub(self, rhs: Q32) -> Option<Q32> {
        match self.0.checked_sub(rhs.0) {
            Some(bits) => Some(Q32(bits)),
            None => None,
        }
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

    /// `np.arctan`: `atan(t) = t * P(t^2)` on `[-1, 1]` (the polynomial of
    /// [`Q32::asin`]), and `pi/2 - atan(1/|x|)` outside, with the sign of `x`.
    pub fn atan(self) -> Q32 {
        let magnitude = self.abs();
        let angle = if magnitude <= Self::ONE {
            magnitude * horner(magnitude * magnitude, &ATAN_OVER_X)
        } else {
            let t = Self::ONE / magnitude;
            Self::FRAC_PI_2 - t * horner(t * t, &ATAN_OVER_X)
        };
        if self.is_negative() {
            -angle
        } else {
            angle
        }
    }

    /// `np.arctan2(self, x)`: the angle of the point `(x, self)` in
    /// `[-pi, pi]`, from [`Q32::atan`] of the smaller over the larger
    /// magnitude (the ratio stays in `[0, 1]`, so it cannot overflow).
    /// `arctan2(0, 0)` is 0, as in NumPy.
    pub fn atan2(self, x: Q32) -> Q32 {
        let (ay, ax) = (self.abs(), x.abs());
        if ay == Self::ZERO && ax == Self::ZERO {
            return Self::ZERO;
        }
        let first_quadrant = if ay <= ax {
            (ay / ax).atan()
        } else {
            Self::FRAC_PI_2 - (ax / ay).atan()
        };
        let upper_half = if x.is_negative() { Self::PI - first_quadrant } else { first_quadrant };
        if self.is_negative() {
            -upper_half
        } else {
            upper_half
        }
    }

    /// `np.exp`: write `x = k * ln 2 + r` with `|r| <= ln(2)/2`, evaluate the
    /// Taylor polynomial of `exp(r)` (degree 13, truncation error below 1e-13
    /// relative), and scale by `2^k`. Zinnia's halo2-base backend has no
    /// `exp`; this is the textbook fixed-point algorithm. Panics when the
    /// result does not fit Q32 (`x` above about 21.4, where NumPy still has
    /// a finite float).
    pub fn exp(self) -> Q32 {
        let k = (self * Self::LOG2_E).round_to_int();
        let r = self - Self::LN_2 * k;
        let value = horner(r, &EXP_TAYLOR);
        match k {
            k if k >= 0 => {
                let scaled = (value.0 as i128) << k;
                Q32(i64::try_from(scaled).expect("exp overflows Q32"))
            }
            k if k > -64 => Q32(value.0 >> -k),
            _ => Q32::ZERO,
        }
    }

    /// `np.log` of a positive value: write `x = m * 2^k` with `m` in
    /// `[1, 2)`, and `ln m = 2 * atanh(s) = 2s(1 + s^2/3 + s^4/5 + ...)` with
    /// `s = (m - 1)/(m + 1)` in `[0, 1/3)` (11 terms, truncation below 1e-12).
    /// Zinnia's halo2-base backend has no `log`; this is the textbook
    /// fixed-point algorithm.
    pub fn ln(self) -> Q32 {
        assert!(self.0 > 0, "log of a non-positive number");
        let k = 63 - self.0.leading_zeros() as i64 - FRAC_BITS as i64;
        let m = Q32(if k >= 0 { self.0 >> k } else { self.0 << -k });
        let s = (m - Self::ONE) / (m + Self::ONE);
        let series = s * horner(s * s, &LN_ODD_SERIES);
        series + series + Self::LN_2 * k
    }

    /// `np.tanh`: `(1 - e) / (1 + e)` with `e = exp(-2|x|)` in `(0, 1]` (no
    /// overflow), with the sign of `x`.
    pub fn tanh(self) -> Q32 {
        let e = (self.abs() * -2).exp();
        let value = (Self::ONE - e) / (Self::ONE + e);
        if self.is_negative() {
            -value
        } else {
            value
        }
    }

    /// `x ** n` for a non-negative integer `n` by repeated multiplication,
    /// rescaling after each product (Zinnia `pow_f` with an integer
    /// exponent).
    pub fn powi(self, n: u32) -> Q32 {
        (0..n).fold(Self::ONE, |acc, _| acc * self)
    }

    /// `x ** y` for `x > 0`: `exp(y * ln x)`.
    pub fn powf(self, y: Q32) -> Q32 {
        (y * self.ln()).exp()
    }

    /// The nearest integer, halves away from zero (`round` in Python 2 /
    /// C; used for argument reduction).
    fn round_to_int(self) -> i64 {
        let half = SCALE / 2;
        if self.0 >= 0 {
            (self.0 + half) >> FRAC_BITS
        } else {
            -((-self.0 + half) >> FRAC_BITS)
        }
    }
}

impl Q32 {
    /// `round(ln 2 * 2^32)`.
    pub const LN_2: Q32 = Q32(2_977_044_472);
    /// `round(log2(e) * 2^32)`.
    pub const LOG2_E: Q32 = Q32(6_196_328_019);
}

/// `round(c * 2^32)` of `1/k!` for k = 13 down to 0: Taylor coefficients of
/// `exp(r)`.
const EXP_TAYLOR: [Q32; 14] = [
    Q32(1),
    Q32(9),
    Q32(108),
    Q32(1_184),
    Q32(11_836),
    Q32(106_522),
    Q32(852_176),
    Q32(5_965_232),
    Q32(35_791_394),
    Q32(178_956_971),
    Q32(715_827_883),
    Q32(2_147_483_648),
    Q32(4_294_967_296),
    Q32(4_294_967_296),
];

/// `round(c * 2^32)` of `1/(2k+1)` for k = 10 down to 0: the series of
/// `atanh(s)/s` in `u = s^2`.
const LN_ODD_SERIES: [Q32; 11] = [
    Q32(204_522_252),
    Q32(226_050_910),
    Q32(252_645_135),
    Q32(286_331_153),
    Q32(330_382_100),
    Q32(390_451_572),
    Q32(477_218_588),
    Q32(613_566_757),
    Q32(858_993_459),
    Q32(1_431_655_765),
    Q32(4_294_967_296),
];

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
        Q32(mul_rescaled(self.0, rhs.0))
    }
}

/// `floor(a * b / 2^32)`, wrapped to 64 bits.
///
/// On 64-bit targets (SP1, the native runner) this is one `i128` product and
/// a shift. On 32-bit targets (RISC Zero) it is [`mul_halves`]: the RISC Zero
/// toolchain (rustc 1.97, riscv32im) miscompiles a chain of such `i128`
/// products. In pythran d2q9, `1.5 * m * m` came out as
/// `m * (lo32(m) - 1) / 2^32`; the 64-bit arithmetic of `mul_halves` avoids
/// the miscompiled pattern. The two forms agree on every input
/// (test `mul_halves_is_the_i128_product`).
#[inline]
fn mul_rescaled(a: i64, b: i64) -> i64 {
    #[cfg(target_pointer_width = "32")]
    {
        mul_halves(a, b)
    }
    #[cfg(not(target_pointer_width = "32"))]
    {
        ((a as i128 * b as i128) >> FRAC_BITS) as i64
    }
}

/// `floor(a * b / 2^32)` from the 32-bit halves `a = a_hi * 2^32 + a_lo`
/// (`a_lo` in `[0, 2^32)`): `a_hi*b_hi*2^32 + a_hi*b_lo + a_lo*b_hi +
/// floor(a_lo*b_lo / 2^32)`, where only the last term has a fraction to
/// drop. The cross products fit an `i64`; the sum wraps like the `as i64` of
/// the `i128` form.
#[cfg_attr(not(target_pointer_width = "32"), allow(dead_code))]
#[inline]
fn mul_halves(a: i64, b: i64) -> i64 {
    const LOW: i64 = 0xffff_ffff;
    let (a_hi, a_lo) = (a >> FRAC_BITS, a & LOW);
    let (b_hi, b_lo) = (b >> FRAC_BITS, b & LOW);
    let low = ((a_lo as u64 * b_lo as u64) >> FRAC_BITS) as i64;
    (a_hi.wrapping_mul(b_hi) << FRAC_BITS)
        .wrapping_add(a_hi * b_lo)
        .wrapping_add(a_lo * b_hi)
        .wrapping_add(low)
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

/// A complex number with Q32 parts (NumPy `complex128` in fixed point).
/// Products accumulate their raw terms and rescale once per part, like
/// [`dot`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct C32 {
    pub re: Q32,
    pub im: Q32,
}

impl C32 {
    pub const ZERO: C32 = C32 { re: Q32::ZERO, im: Q32::ZERO };
    pub const ONE: C32 = C32 { re: Q32::ONE, im: Q32::ZERO };
    /// The imaginary unit `1j`.
    pub const I: C32 = C32 { re: Q32::ZERO, im: Q32::ONE };

    pub const fn new(re: Q32, im: Q32) -> C32 {
        C32 { re, im }
    }

    pub const fn from_real(re: Q32) -> C32 {
        C32 { re, im: Q32::ZERO }
    }

    /// `np.conj`.
    pub fn conj(self) -> C32 {
        C32 { re: self.re, im: -self.im }
    }

    /// `|z|^2 = re^2 + im^2`, rescaled once.
    pub fn norm_sqr(self) -> Q32 {
        dot([(self.re, self.re), (self.im, self.im)])
    }

    /// `np.abs`: `floor(sqrt((re^2 + im^2) * 2^32))` from the exact sum of
    /// squares.
    pub fn abs(self) -> Q32 {
        let raw = (self.re.0 as i128).pow(2) + (self.im.0 as i128).pow(2);
        Q32(i64::try_from((raw as u128).isqrt()).expect("abs overflows Q32"))
    }

    /// `np.exp` of a complex number: `e^re * (cos im + i sin im)`.
    pub fn exp(self) -> C32 {
        let magnitude = self.re.exp();
        C32 { re: magnitude * self.im.cos(), im: magnitude * self.im.sin() }
    }
}

impl Add for C32 {
    type Output = C32;
    fn add(self, rhs: C32) -> C32 {
        C32 { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for C32 {
    type Output = C32;
    fn sub(self, rhs: C32) -> C32 {
        C32 { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl Neg for C32 {
    type Output = C32;
    fn neg(self) -> C32 {
        C32 { re: -self.re, im: -self.im }
    }
}

/// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, each part rescaled once.
impl Mul for C32 {
    type Output = C32;
    fn mul(self, rhs: C32) -> C32 {
        C32 {
            re: dot([(self.re, rhs.re), (-self.im, rhs.im)]),
            im: dot([(self.re, rhs.im), (self.im, rhs.re)]),
        }
    }
}

/// Scaling by a real number.
impl Mul<Q32> for C32 {
    type Output = C32;
    fn mul(self, rhs: Q32) -> C32 {
        C32 { re: self.re * rhs, im: self.im * rhs }
    }
}

/// `a / b = a * conj(b) / |b|^2`; panics if `b == 0`.
impl Div for C32 {
    type Output = C32;
    fn div(self, rhs: C32) -> C32 {
        let numerator = self * rhs.conj();
        let denominator = rhs.norm_sqr();
        C32 { re: numerator.re / denominator, im: numerator.im / denominator }
    }
}

/// Division by a real number.
impl Div<Q32> for C32 {
    type Output = C32;
    fn div(self, rhs: Q32) -> C32 {
        C32 { re: self.re / rhs, im: self.im / rhs }
    }
}

impl AddAssign for C32 {
    fn add_assign(&mut self, rhs: C32) {
        *self = *self + rhs;
    }
}

impl SubAssign for C32 {
    fn sub_assign(&mut self, rhs: C32) {
        *self = *self - rhs;
    }
}

impl MulAssign for C32 {
    fn mul_assign(&mut self, rhs: C32) {
        *self = *self * rhs;
    }
}

impl core::iter::Sum for C32 {
    fn sum<I: Iterator<Item = C32>>(iter: I) -> C32 {
        iter.fold(C32::ZERO, Add::add)
    }
}

impl From<Q32> for C32 {
    fn from(re: Q32) -> C32 {
        C32::from_real(re)
    }
}

/// `ndarray` support: `Array::zeros`/`ones` (num-traits `Zero`/`One`) and
/// array-by-scalar arithmetic (`ScalarOperand`).
#[cfg(feature = "ndarray")]
mod ndarray_support {
    use super::{C32, Q32};

    impl num_traits::Zero for Q32 {
        fn zero() -> Q32 {
            Q32::ZERO
        }
        fn is_zero(&self) -> bool {
            *self == Q32::ZERO
        }
    }

    impl num_traits::One for Q32 {
        fn one() -> Q32 {
            Q32::ONE
        }
    }

    impl num_traits::Zero for C32 {
        fn zero() -> C32 {
            C32::ZERO
        }
        fn is_zero(&self) -> bool {
            *self == C32::ZERO
        }
    }

    impl num_traits::One for C32 {
        fn one() -> C32 {
            C32::ONE
        }
    }

    impl ndarray::ScalarOperand for Q32 {}
    impl ndarray::ScalarOperand for C32 {}
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
    fn mul_halves_is_the_i128_product() {
        let i128_form = |a: i64, b: i64| ((a as i128 * b as i128) >> FRAC_BITS) as i64;
        let edges = [
            0, 1, -1, 2, -2, i64::MAX, i64::MIN, i64::MIN + 1, 1 << 31, -(1 << 31), 3 << 31,
            -(3 << 31), 1 << 32, -(1 << 32), (1 << 32) - 1, -(1 << 32) + 1, 0x7fff_ffff, -0x8000_0000,
        ];
        for &a in &edges {
            for &b in &edges {
                assert_eq!(mul_halves(a, b), i128_form(a, b), "{a} * {b}");
            }
        }
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            // Mix magnitudes: full 64-bit values and values within Q32's range.
            let v = state as i64;
            if state >> 63 == 0 { v >> (state % 40) } else { v }
        };
        for _ in 0..1_000_000 {
            let (a, b) = (next(), next());
            assert_eq!(mul_halves(a, b), i128_form(a, b), "{a} * {b}");
        }
    }

    #[test]
    fn exp_log_tanh_atan_match_libm() {
        // exp: relative error, over the whole Q32 range of its result.
        for i in -2000..=2140 {
            let q = Q32::from_f64(i as f64 / 100.0);
            let want = q.to_f64().exp();
            let got = q.exp().to_f64();
            assert!((got - want).abs() <= 2e-9 * want.max(1.0), "exp {q:?}: {got} vs {want}");
        }
        for v in [1e-9, 1e-3, 0.37, 1.0, 1.5, 2.0, 7.25, 1234.5, 2.1e9] {
            let q = Q32::from_f64(v);
            assert!(close(q.ln(), q.to_f64().ln(), 2e-9), "ln {v}");
        }
        for i in -3000..=3000 {
            let q = Q32::from_f64(i as f64 / 100.0);
            assert!(close(q.tanh(), q.to_f64().tanh(), 2e-9), "tanh {q:?}");
            assert!(close(q.atan(), q.to_f64().atan(), 1e-8), "atan {q:?}");
            for x in [-2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5] {
                let qx = Q32::from_f64(x);
                assert!(close(q.atan2(qx), q.to_f64().atan2(qx.to_f64()), 1e-8), "atan2 {q:?} {x}");
            }
        }
        let two = Q32::from_int(2);
        assert!(close(two.powf(Q32::from_f64(0.5)), 2f64.sqrt(), 1e-8));
        assert_eq!(two.powi(10), Q32::from_int(1024));
    }

    #[test]
    fn complex_arithmetic() {
        let a = C32::new(Q32::from_f64(1.5), Q32::from_f64(-2.0));
        let b = C32::new(Q32::from_f64(0.25), Q32::from_f64(3.0));
        let product = a * b;
        assert_eq!((product.re.to_f64(), product.im.to_f64()), (6.375, 4.0));
        let quotient = product / b;
        assert!(close(quotient.re, 1.5, 1e-9) && close(quotient.im, -2.0, 1e-9));
        assert_eq!(a.conj(), C32::new(a.re, -a.im));
        assert!(close(C32::new(Q32::from_int(3), Q32::from_int(4)).abs(), 5.0, 1e-12));
        let e = C32::new(Q32::ZERO, Q32::PI).exp();
        assert!(close(e.re, -1.0, 1e-8) && close(e.im, 0.0, 1e-8));
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
    fn checked_abs_and_sub_stay_in_q32() {
        assert_eq!(Q32::from_bits(-5).checked_abs(), Some(Q32::from_bits(5)));
        assert_eq!(Q32::from_bits(i64::MIN + 1).checked_abs(), Some(Q32::from_bits(i64::MAX)));
        assert_eq!(Q32::from_bits(i64::MIN).checked_abs(), None);
        assert_eq!(Q32::ONE.checked_sub(Q32::HALF), Some(Q32::HALF));
        assert_eq!(Q32::from_bits(i64::MIN).checked_sub(Q32::from_bits(1)), None);
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
