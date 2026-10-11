//! Q32.32 fixed-point numbers: the same encoding and rounding as Zinnia and as
//! the Rust zkVM baselines (`zkvm_common/q32`).
//!
//! A real value `v` is the integer `round(v * 2^32)`, held in an `i64`.
//! * `a * b` is `floor(a * b / 2^32)` (Zinnia `fp_mul_rescale`);
//! * `a / b` is `floor(a * 2^32 / b)` with Euclidean rounding (Zinnia
//!   `fp_div_prescale`);
//! * `sqrt` is `floor(sqrt(a * 2^32))`, the exact integer root;
//! * `sin`/`cos`/`asin` are Zinnia's fixed-point algorithms (`fp_sin`,
//!   `fp_cos`, `fp_arcsin`, which the halo2-base circuit constrains): range
//!   reduction to `[0, pi/4]` with Taylor polynomials in `s^2`, and
//!   `2 * atan(x / (1 + sqrt(1 - x^2)))`; Horner rescales every product.
use core::num::traits::Sqrt;

pub const HALF: Q32 = Q32 { bits: 2147483648 };
pub const LN_2: Q32 = Q32 { bits: 2977044472 };
pub const LOG2_E: Q32 = Q32 { bits: 6196328019 };
const EXP_TAYLOR: [i64; 14] = [1, 9, 108, 1184, 11836, 106522, 852176, 5965232,
    35791394, 178956971, 715827883, 2147483648, 4294967296, 4294967296];
const LN_ODD_SERIES: [i64; 11] = [204522252, 226050910, 252645135, 286331153,
    330382100, 390451572, 477218588, 613566757, 858993459, 1431655765, 4294967296];

const SCALE: i128 = 0x100000000;

#[derive(Copy, Drop, Serde, PartialEq, Debug)]
pub struct Q32 {
    pub bits: i64,
}

pub const ZERO: Q32 = Q32 { bits: 0 };
pub const ONE: Q32 = Q32 { bits: 0x100000000 };
/// `round(pi * 2^32)`.
pub const PI: Q32 = Q32 { bits: 13493037705 };
/// `round(pi / 2 * 2^32)`.
pub const FRAC_PI_2: Q32 = Q32 { bits: 6746518852 };
/// `round(pi / 180 * 2^32)`, the factor of `np.deg2rad`.
pub const DEG_TO_RAD: Q32 = Q32 { bits: 74961321 };
/// `round(180 / pi * 2^32)`, the factor of `np.degrees`.
pub const RAD_TO_DEG: Q32 = Q32 { bits: 246083499208 };

/// `round(pi / 4 * 2^32)`.
pub const FRAC_PI_4: Q32 = Q32 { bits: 3373259426 };

/// `round(c * 2^32)` of `(-1)^k / (2k+1)!` for k = 6 down to 0: Taylor
/// coefficients of `sin(s) / s` in `u = s^2` (Zinnia `sin_taylor_q`).
const SIN_OVER_X_TAYLOR: [i64; 7] = [1, -108, 11836, -852176, 35791394, -715827883, 4294967296];

/// `round(c * 2^32)` of `(-1)^k / (2k)!` for k = 7 down to 0: Taylor
/// coefficients of `cos(s)` in `u = s^2` (Zinnia `cos_taylor_q`).
const COS_TAYLOR: [i64; 8] = [0, 9, -1184, 106522, -5965232, 178956971, -2147483648, 4294967296];

/// `round(c * 2^32)` of Zinnia's `ATAN_ODD_U_COEFS` (`atan(t) / t` as a
/// degree-9 polynomial in `u = t^2`), highest degree first.
const ATAN_OVER_X: [i64; 10] = [
    -7489138, 45873335, -131930687, 246804093, -359577404, 469877966, -612529925, 858917466,
    -1431653573, 4294967285,
];

/// `floor(a / b)`; Cairo's `/` on signed integers truncates toward zero.
pub fn floor_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Division with a remainder in `[0, |b|)`.
fn euclid_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a % b < 0 {
        if b > 0 {
            q - 1
        } else {
            q + 1
        }
    } else {
        q
    }
}

/// The unscaled product of two Q32 values, for sums of products that are
/// rescaled once (Zinnia `fp_product_sum_rescale`).
pub fn raw_product(a: Q32, b: Q32) -> i128 {
    let x: i128 = a.bits.into();
    x * b.bits.into()
}

/// Rescales a sum of raw products: `floor(raw / 2^32)`.
pub fn from_raw_product(raw: i128) -> Q32 {
    Q32 { bits: floor_div(raw, SCALE).try_into().unwrap() }
}

#[generate_trait]
pub impl Q32Impl of Q32Trait {
    fn floor(self: Q32) -> Q32 {
        Q32 { bits: (floor_div(self.bits.into(), SCALE) * SCALE).try_into().unwrap() }
    }

    fn atan(self: Q32) -> Q32 {
        let magnitude = self.abs();
        let angle = if magnitude <= ONE {
            magnitude * horner(magnitude * magnitude, ATAN_OVER_X.span())
        } else {
            let t = ONE / magnitude;
            FRAC_PI_2 - t * horner(t * t, ATAN_OVER_X.span())
        };
        if self.bits < 0 { -angle } else { angle }
    }

    fn atan2(self: Q32, x: Q32) -> Q32 {
        let ay = self.abs();
        let ax = x.abs();
        if ay == ZERO && ax == ZERO { return ZERO; }
        let first = if ay <= ax { (ay / ax).atan() } else { FRAC_PI_2 - (ax / ay).atan() };
        let upper = if x.bits < 0 { PI - first } else { first };
        if self.bits < 0 { -upper } else { upper }
    }

    /// Rust's degree-13 Taylor expansion after nearest-integer range reduction.
    fn exp(self: Q32) -> Q32 {
        let scaled = self * LOG2_E;
        let k = if scaled.bits >= 0 {
            (scaled.bits + HALF.bits) / ONE.bits
        } else { -((-scaled.bits + HALF.bits) / ONE.bits) };
        let r = self - LN_2.mul_int(k);
        let value = horner(r, EXP_TAYLOR.span());
        if k >= 0 {
            // A positive exponent of 32 or more cannot fit Q32.
            assert!(k < 32, "exp overflows Q32");
            let factor = pow_two(k.try_into().unwrap());
            Q32 { bits: (Into::<i64, i128>::into(value.bits) * factor).try_into().unwrap() }
        } else if k > -64 {
            Q32 { bits: (Into::<i64, i128>::into(value.bits) / pow_two((-k).try_into().unwrap())).try_into().unwrap() }
        } else { ZERO }
    }

    fn ln(self: Q32) -> Q32 {
        assert!(self.bits > 0, "log of a non-positive number");
        let mut top = self.bits;
        let mut k: i64 = -32;
        while top >= 2 { top /= 2; k += 1; };
        let m = if k >= 0 {
            Q32 { bits: (Into::<i64, i128>::into(self.bits) / pow_two(k.try_into().unwrap())).try_into().unwrap() }
        } else {
            Q32 { bits: (Into::<i64, i128>::into(self.bits) * pow_two((-k).try_into().unwrap())).try_into().unwrap() }
        };
        let s = (m - ONE) / (m + ONE);
        let series = s * horner(s * s, LN_ODD_SERIES.span());
        series + series + LN_2.mul_int(k)
    }

    fn tanh(self: Q32) -> Q32 {
        let e = self.abs().mul_int(-2).exp();
        let value = (ONE - e) / (ONE + e);
        if self.bits < 0 { -value } else { value }
    }

    /// Deliberately rescale every successive product, as Rust does.
    fn powi(self: Q32, n: u32) -> Q32 {
        let mut acc = ONE;
        for _ in 0..n { acc = acc * self; };
        acc
    }

    fn powf(self: Q32, y: Q32) -> Q32 { (y * self.ln()).exp() }

    /// The exact value of an integer (Zinnia `float_cast`).
    fn from_int(n: i64) -> Q32 {
        Q32 { bits: n * 0x100000000 }
    }

    /// Python `int(x)`: truncate toward zero.
    fn trunc_to_int(self: Q32) -> i64 {
        self.bits / 0x100000000
    }

    fn abs(self: Q32) -> Q32 {
        if self.bits < 0 {
            Q32 { bits: -self.bits }
        } else {
            self
        }
    }

    /// Exact multiplication by an integer.
    fn mul_int(self: Q32, n: i64) -> Q32 {
        Q32 { bits: self.bits * n }
    }

    /// Division by an integer, equal to `self / from_int(n)`.
    fn div_int(self: Q32, n: i64) -> Q32 {
        Q32 { bits: euclid_div(self.bits.into(), n.into()).try_into().unwrap() }
    }

    /// `np.sqrt` of a non-negative value: `floor(sqrt(x * 2^32))`.
    fn sqrt(self: Q32) -> Q32 {
        assert!(self.bits >= 0, "sqrt of a negative number");
        let as_felt: felt252 = self.bits.into();
        let x: u128 = as_felt.try_into().unwrap();
        let root: u64 = (x * 0x100000000).sqrt();
        Q32 { bits: root.try_into().unwrap() }
    }

    /// `np.sin` (Zinnia `fp_sin`): reduce `|x|` modulo `2*pi`, fold onto
    /// `[0, pi)` (`sin(r) = -sin(r - pi)`), mirror onto `[0, pi/2]`
    /// (`sin(r) = sin(pi - r)`), and above `pi/4` use `cos(pi/2 - r)`, so the
    /// Taylor polynomials run on `s` in `[0, pi/4]`, in `u = s^2`.
    fn sin(self: Q32) -> Q32 {
        let r = self.abs().bits % (2 * PI.bits);
        let (r, folded) = if r >= PI.bits {
            (r - PI.bits, true)
        } else {
            (r, false)
        };
        let r = if r > FRAC_PI_2.bits {
            PI.bits - r
        } else {
            r
        };
        let value = if r > FRAC_PI_4.bits {
            let s = Q32 { bits: FRAC_PI_2.bits - r };
            horner(s * s, COS_TAYLOR.span())
        } else {
            let s = Q32 { bits: r };
            s * horner(s * s, SIN_OVER_X_TAYLOR.span())
        };
        if (self.bits < 0) != folded {
            -value
        } else {
            value
        }
    }

    /// `np.cos(x) = sin(x + pi/2)` (Zinnia `fp_cos`).
    fn cos(self: Q32) -> Q32 {
        (self + FRAC_PI_2).sin()
    }

    /// `np.arcsin` for `x` in `[-1, 1]` (Zinnia `fp_arcsin`):
    /// `2 * atan(x / (1 + sqrt(1 - x^2)))`. The root is the exact integer root
    /// of `2^64 - x^2` in raw units, and `atan(t) = t * P(t^2)` with `t` in
    /// `[-1, 1]`.
    fn asin(self: Q32) -> Q32 {
        assert!(-ONE <= self && self <= ONE, "arcsin argument outside [-1, 1]");
        let x: i128 = self.bits.into();
        let one: i128 = ONE.bits.into();
        let radicand: u128 = (one * one - x * x).try_into().unwrap();
        let root: u64 = radicand.sqrt();
        let t = self / Q32 { bits: ONE.bits + root.try_into().unwrap() };
        let half = t * horner(t * t, ATAN_OVER_X.span());
        half + half
    }
}

/// Horner's rule, highest degree first, rescaling every product.
fn horner(x: Q32, coefs: Span<i64>) -> Q32 {
    let mut acc = Q32 { bits: *coefs.at(0) };
    for i in 1..coefs.len() {
        acc = acc * x + Q32 { bits: *coefs.at(i) };
    }
    acc
}

impl Q32Add of Add<Q32> {
    fn add(lhs: Q32, rhs: Q32) -> Q32 {
        Q32 { bits: lhs.bits + rhs.bits }
    }
}

impl Q32Sub of Sub<Q32> {
    fn sub(lhs: Q32, rhs: Q32) -> Q32 {
        Q32 { bits: lhs.bits - rhs.bits }
    }
}

impl Q32Neg of Neg<Q32> {
    fn neg(a: Q32) -> Q32 {
        Q32 { bits: -a.bits }
    }
}

/// `floor(a * b / 2^32)`.
impl Q32Mul of Mul<Q32> {
    fn mul(lhs: Q32, rhs: Q32) -> Q32 {
        from_raw_product(raw_product(lhs, rhs))
    }
}

/// `floor(a * 2^32 / b)`, Euclidean.
impl Q32Div of Div<Q32> {
    fn div(lhs: Q32, rhs: Q32) -> Q32 {
        let a: i128 = lhs.bits.into();
        Q32 { bits: euclid_div(a * SCALE, rhs.bits.into()).try_into().unwrap() }
    }
}

impl Q32PartialOrd of PartialOrd<Q32> {
    fn lt(lhs: Q32, rhs: Q32) -> bool {
        lhs.bits < rhs.bits
    }
    fn le(lhs: Q32, rhs: Q32) -> bool {
        lhs.bits <= rhs.bits
    }
    fn gt(lhs: Q32, rhs: Q32) -> bool {
        lhs.bits > rhs.bits
    }
    fn ge(lhs: Q32, rhs: Q32) -> bool {
        lhs.bits >= rhs.bits
    }
}

#[cfg(test)]
mod tests {
    use super::{ONE, Q32, Q32Trait};

    #[test]
    fn rescaling_floors() {
        let eps = Q32 { bits: 1 };
        let half = Q32 { bits: 0x80000000 };
        assert!(half * eps == Q32 { bits: 0 });
        assert!(-half * eps == Q32 { bits: -1 });
        assert!(-ONE / Q32Trait::from_int(3) == Q32 { bits: -1431655766 });
        assert!(Q32Trait::from_int(-7).div_int(2) == Q32 { bits: -15032385536 });
    }
}

/// Small exact powers used for fixed-point range reduction.
fn pow_two(n: u32) -> i128 {
    let mut value: i128 = 1;
    for _ in 0..n { value *= 2; };
    value
}

/// Complex Q32 arithmetic; each product component is rescaled only once.
#[derive(Copy, Drop, Serde, PartialEq, Debug)]
pub struct C32 { pub re: Q32, pub im: Q32 }
pub const C_ZERO: C32 = C32 { re: ZERO, im: ZERO };
pub const C_ONE: C32 = C32 { re: ONE, im: ZERO };
pub const I: C32 = C32 { re: ZERO, im: ONE };

#[generate_trait]
pub impl C32Impl of C32Trait {
    fn new(re: Q32, im: Q32) -> C32 { C32 { re, im } }
    fn from_real(re: Q32) -> C32 { C32 { re, im: ZERO } }
    fn conj(self: C32) -> C32 { C32 { re: self.re, im: -self.im } }
    fn norm_sqr(self: C32) -> Q32 {
        from_raw_product(raw_product(self.re, self.re) + raw_product(self.im, self.im))
    }
    fn abs(self: C32) -> Q32 {
        let raw: u128 = (raw_product(self.re, self.re) + raw_product(self.im, self.im)).try_into().unwrap();
        let root: u64 = raw.sqrt();
        Q32 { bits: root.try_into().unwrap() }
    }
    fn exp(self: C32) -> C32 {
        let magnitude = self.re.exp();
        C32 { re: magnitude * self.im.cos(), im: magnitude * self.im.sin() }
    }
    fn scale(self: C32, rhs: Q32) -> C32 { C32 { re: self.re * rhs, im: self.im * rhs } }
    fn div_real(self: C32, rhs: Q32) -> C32 { C32 { re: self.re / rhs, im: self.im / rhs } }
}
impl C32Add of Add<C32> {
    fn add(lhs: C32, rhs: C32) -> C32 { C32 { re: lhs.re + rhs.re, im: lhs.im + rhs.im } }
}
impl C32Sub of Sub<C32> {
    fn sub(lhs: C32, rhs: C32) -> C32 { C32 { re: lhs.re - rhs.re, im: lhs.im - rhs.im } }
}
impl C32Neg of Neg<C32> {
    fn neg(a: C32) -> C32 { C32 { re: -a.re, im: -a.im } }
}
impl C32Mul of Mul<C32> {
    fn mul(lhs: C32, rhs: C32) -> C32 {
        C32 {
            re: from_raw_product(raw_product(lhs.re, rhs.re) + raw_product(-lhs.im, rhs.im)),
            im: from_raw_product(raw_product(lhs.re, rhs.im) + raw_product(lhs.im, rhs.re)),
        }
    }
}
impl C32Div of Div<C32> {
    fn div(lhs: C32, rhs: C32) -> C32 {
        (lhs * rhs.conj()).div_real(rhs.norm_sqr())
    }
}

#[cfg(test)]
mod rust_vectors;

#[cfg(test)]
mod domain_tests {
    use super::{Q32, Q32Trait, ZERO};
    #[test]
    #[should_panic]
    fn log_zero_panics() { let _ = ZERO.ln(); }
    #[test]
    #[should_panic]
    fn log_negative_panics() { let _ = Q32 { bits: -1 }.ln(); }
    #[test]
    #[should_panic]
    fn exp_overflow_panics() { let _ = Q32Trait::from_int(22).exp(); }
    #[test]
    #[should_panic]
    fn negative_fractional_power_panics() { let _ = Q32 { bits: -1 }.powf(ZERO); }
}
