//! Q32.32 fixed-point numbers: the same encoding and rounding as Zinnia and as
//! the Rust zkVM baselines (`zkvm_common/q32`).
//!
//! A real value `v` is the integer `round(v * 2^32)`, held in an `i64`.
//! * `a * b` is `floor(a * b / 2^32)` (Zinnia `fp_mul_rescale`);
//! * `a / b` is `floor(a * 2^32 / b)` with Euclidean rounding (Zinnia
//!   `fp_div_prescale`);
//! * `sqrt` is `floor(sqrt(a * 2^32))`;
//! * `sin`/`cos` are Zinnia's halo2-base definition (reduce modulo pi, degree-14
//!   Remez polynomial, Horner with a rescale per step);
//! * `asin` is `atan2(x, sqrt(1 - x^2))` with the degree-15 atan polynomial.
use core::num::traits::Sqrt;

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

/// `round(c * 2^32)` of Zinnia's `SIN_COEFS` (sin on `[0, pi)`), highest
/// degree first.
const SIN_COEFS: [i64; 15] = [
    0, 1, -2, -102, -13, 11858, -28, -852151, -16, 35791402, -2, -715827882, 0, 4294967296, 0,
];

/// `round(c * 2^32)` of Zinnia's `ATAN_COEFS` (atan on `[-1, 1]`), highest
/// degree first.
const ATAN_COEFS: [i64; 16] = [
    -17009175, 0, 92381544, 0, -237862581, 0, 412388002, 0, -596658626, 0, 856548815, 0,
    -1431492759, 0, 4294964071, 0,
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

    /// `np.sin` (Zinnia halo2-base `sin_f`).
    fn sin(self: Q32) -> Q32 {
        let magnitude = self.abs().bits;
        let half_turns = magnitude / PI.bits;
        let reduced = Q32 { bits: magnitude % PI.bits };
        let value = horner(reduced, SIN_COEFS.span());
        if (self.bits < 0) != (half_turns % 2 == 1) {
            -value
        } else {
            value
        }
    }

    /// `np.cos(x) = sin(x + pi/2)` (Zinnia halo2-base `cos_f`).
    fn cos(self: Q32) -> Q32 {
        (self + FRAC_PI_2).sin()
    }

    /// `np.arcsin` for `x` in `[-1, 1]`.
    fn asin(self: Q32) -> Q32 {
        assert!(-ONE <= self && self <= ONE, "arcsin argument outside [-1, 1]");
        atan2(self, (ONE - self * self).sqrt())
    }
}

/// `np.arctan2(y, x)` via the atan polynomial on a ratio in `[0, 1]`.
pub fn atan2(y: Q32, x: Q32) -> Q32 {
    let ay = y.abs();
    let ax = x.abs();
    if ay == ZERO && ax == ZERO {
        return ZERO;
    }
    let steep = ay > ax;
    let ratio = if steep {
        ax / ay
    } else {
        ay / ax
    };
    let base = horner(ratio, ATAN_COEFS.span());
    let first_quadrant = if steep {
        FRAC_PI_2 - base
    } else {
        base
    };
    let upper_half = if x.bits < 0 {
        PI - first_quadrant
    } else {
        first_quadrant
    };
    if y.bits < 0 {
        -upper_half
    } else {
        upper_half
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

/// Stable merge sort (Cairo arrays are append-only, so sorting builds new
/// arrays).
pub fn sort<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(values: Span<T>) -> Array<T> {
    if values.len() <= 1 {
        let mut out = array![];
        for v in values {
            out.append(*v);
        }
        return out;
    }
    let mid = values.len() / 2;
    let left = sort(values.slice(0, mid));
    let right = sort(values.slice(mid, values.len() - mid));
    merge(left.span(), right.span())
}

fn merge<T, +Copy<T>, +Drop<T>, +PartialOrd<T>>(left: Span<T>, right: Span<T>) -> Array<T> {
    let mut out = array![];
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        if *right.at(j) < *left.at(i) {
            out.append(*right.at(j));
            j += 1;
        } else {
            out.append(*left.at(i));
            i += 1;
        }
    }
    while i < left.len() {
        out.append(*left.at(i));
        i += 1;
    }
    while j < right.len() {
        out.append(*right.at(j));
        j += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{ONE, Q32, Q32Trait, sort};

    #[test]
    fn rescaling_floors() {
        let eps = Q32 { bits: 1 };
        let half = Q32 { bits: 0x80000000 };
        assert!(half * eps == Q32 { bits: 0 });
        assert!(-half * eps == Q32 { bits: -1 });
        assert!(-ONE / Q32Trait::from_int(3) == Q32 { bits: -1431655766 });
        assert!(Q32Trait::from_int(-7).div_int(2) == Q32 { bits: -15032385536 });
    }

    #[test]
    fn sorts() {
        let sorted = sort(array![3_i64, -1, 2, -1, 0].span());
        assert!(sorted == array![-1, -1, 0, 2, 3]);
    }
}
