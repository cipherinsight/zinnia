//! Shared computation kernel for all proving backends.
//!
//! This module provides pure functions for field arithmetic, fixed-point
//! quantization, and transcendental function evaluation that are used
//! identically by both the mock and halo2 synthesizers. This ensures
//! semantic consistency: the mock backend produces the exact same values
//! that the halo2 backend constrains.
//!
//! All functions operate on [`crate::prove::field::F`] — BN254's scalar field
//! (`halo2curves::bn256::Fr`). This is the same `Fr` the halo2 backends
//! constrain, sourced from the light `halo2curves` crate so the shared kernel
//! does not link the full `halo2_proofs` proving system.

use crate::prove::field::F as Fp;
use crate::prove::field::PrimeField;

/// Re-export the Field trait so backends can call `.invert()` etc.
pub use crate::prove::field::Field;

// ---------------------------------------------------------------------------
// Remez polynomial coefficients
// ---------------------------------------------------------------------------

/// exp2(x) on [0, 1), degree 12. (from halo2-graph fixed_point.rs)
pub const EXP2_COEFS: [f64; 13] = [
    3.6240421303547230336183979205877e-11,
    4.1284327467833130245549169910389e-10,
    0.0000000071086385644026346316624185550542,
    0.00000010172297085296590958930245291448,
    0.0000013215904023658396206789543841996,
    0.000015252713316417140696221389106544,
    0.00015403531076657894204857389177279,
    0.0013333558131297097698435464957392,
    0.0096181291078409107025643582456283,
    0.055504108664804181586140094858174,
    0.24022650695910142332414229540187,
    0.69314718055994529934452147700678,
    1.0,
];

/// log2(x) on [2, 4), degree 14.
pub const LOG2_COEFS: [f64; 15] = [
    -3.319586265362338e-08,
    1.4957235315170112e-06,
    -3.1350053389526744e-05,
    0.00040554177582512901,
    -0.0036218342998850703,
    0.023663846121538389,
    -0.11691877183255484,
    0.44524062371564499,
    -1.3195777548208449,
    3.0518128028712077,
    -5.4904626000399528,
    7.6298580090181591,
    -8.1653313719804235,
    7.1389971101896279,
    -3.1937385492842112,
];

/// atan(x) on [-1, 1], degree 15 (Chebyshev least-squares fit).
/// Max absolute error ≈ 4.3e-8 over the input range.
///
/// Unlike the other tables this one is stored **constant-term first**
/// (`ATAN_COEFS[k]` multiplies `x^k`; `ATAN_COEFS[1] ≈ 1`, `[3] ≈ -1/3`).
/// The even-indexed coefficients are near zero (atan is odd). Used by the
/// legacy halo2 backend only; the fixed-point `arcsin` uses the more
/// accurate [`ATAN_ODD_U_COEFS`].
pub const ATAN_COEFS: [f64; 16] = [
    3.8945708106325862e-17,
    9.9999924908801763e-01,
    -9.0995364618408146e-16,
    -3.3329538038525558e-01,
    2.0198596716501640e-14,
    1.9943081188176748e-01,
    -1.3380112780080282e-13,
    -1.3892041201045824e-01,
    4.0128381341663972e-13,
    9.6016563957688830e-02,
    -6.0766088433310415e-13,
    -5.5381697876382932e-02,
    4.5616807587830846e-13,
    2.1509254247070526e-02,
    -1.3555701190010715e-13,
    -3.9602572342678811e-03,
];

/// `atan(t) / t` as a polynomial in `u = t^2`, constant term first
/// (`ATAN_ODD_U_COEFS[k]` multiplies `u^k`), so `atan(t) ≈ t * P(t^2)` on
/// `[-1, 1]`. A degree-9 Chebyshev least-squares fit in `u` (Chebyshev
/// nodes in `t`, float64), maximum error 1.8e-9 for `atan`; every
/// coefficient is below 1 in magnitude, so Horner on `u <= 1` does not
/// amplify rounding. Used by the fixed-point `arcsin`.
pub const ATAN_ODD_U_COEFS: [f64; 10] = [
    0.9999999975460914,
    -0.33333282295490435,
    0.19998230639762504,
    -0.14261573673486158,
    0.10940198927495892,
    -0.0837206383142983,
    0.0574635557348283,
    -0.03071750667979075,
    0.010680718187754413,
    -0.0017437008481290968,
];

/// `sin(s) / s` as a polynomial in `u = s^2`, constant term first
/// (`SIN_ODD_U_COEFS[k]` multiplies `u^k`), so `sin(s) ≈ s * P(s^2)` on
/// `[0, π/2]`. A degree-4 least-squares fit in `u`, weighted by `s` so that
/// the absolute error of `s * P(s^2)` is minimized (Chebyshev nodes in `s`,
/// float64). With Q32 coefficients and every product rounded down, the
/// fixed-point `sin` built on it ([`fp_sin`]) is within 9.0e-9 of libm on
/// `[-20, 20]`. Used by the fixed-point `sin`/`cos`.
pub const SIN_ODD_U_COEFS: [f64; 5] = [
    0.9999999812099671,
    -0.16666649686521623,
    0.008332926697684203,
    -0.0001980225217649012,
    2.5928110410078364e-06,
];

/// sin(x) on [0, pi), degree 14 (legacy halo2 backend only).
pub const SIN_COEFS: [f64; 15] = [
    -1.1008071636607462e-11,
    2.4208013888629323e-10,
    -3.8584805817996712e-10,
    -2.3786993104309845e-08,
    -2.9795813710683115e-09,
    2.7608543130047009e-06,
    -6.4467066994122565e-09,
    -0.00019840680551418068,
    -3.839555844512214e-09,
    0.0083333350601673614,
    -5.0943769725466814e-10,
    -0.16666666657583049,
    -8.5029878414113731e-12,
    1.0000000000003146,
    -1.9323057584419828e-15,
];

// ---------------------------------------------------------------------------
// Field element helpers
// ---------------------------------------------------------------------------

/// Convert a signed i64 to an Fp field element.
pub fn i64_to_fp(v: i64) -> Fp {
    if v >= 0 {
        Fp::from(v as u64)
    } else {
        -Fp::from(v.unsigned_abs())
    }
}

/// Convert an Fp field element to a signed i64 (interpreting large values as negative).
///
/// Uses the convention that values > (p-1)/2 represent negative numbers.
pub fn fp_to_i64(v: Fp) -> i64 {
    if v == Fp::zero() {
        return 0;
    }
    let is_neg = fp_is_negative(v);
    if is_neg {
        let pos = -v; // pos = p - v, which is the absolute value
        let pos_bytes = pos.to_repr();
        let low = u64::from_le_bytes(pos_bytes.as_ref()[0..8].try_into().unwrap());
        // The magnitude of i64::MIN is representable only as an unsigned
        // integer; negate there before interpreting the signed bit pattern.
        low.wrapping_neg() as i64
    } else {
        let bytes = v.to_repr();
        let low = u64::from_le_bytes(bytes.as_ref()[0..8].try_into().unwrap());
        low as i64
    }
}

/// Check if a field element represents a negative value (> (p-1)/2).
///
/// The BN254 scalar field has p ≈ 2^254. The "negative" half is [(p+1)/2, p-1].
/// We compute (p-1)/2 and compare using big-endian byte ordering.
pub fn fp_is_negative(v: Fp) -> bool {
    if v == Fp::zero() {
        return false;
    }
    // half = (p-1)/2. In the field, -1 = p-1, so (p-1)/2 = (-1) * 2^{-1}.
    let half = (Fp::zero() - Fp::one()) * Fp::from(2).invert().unwrap();
    // to_repr() returns little-endian bytes. Compare as big-endian for correct ordering.
    let v_bytes = v.to_repr();
    let h_bytes = half.to_repr();
    let v_slice = v_bytes.as_ref();
    let h_slice = h_bytes.as_ref();
    // Compare from most significant byte to least
    for i in (0..v_slice.len()).rev() {
        if v_slice[i] > h_slice[i] {
            return true;
        }
        if v_slice[i] < h_slice[i] {
            return false;
        }
    }
    false // equal to half, treat as non-negative
}

/// The signed integer a field element encodes, if its magnitude is below
/// 2^126 (the window every fixed-point intermediate of the kernel lives in).
/// `None` for anything larger, so callers never silently truncate.
pub fn fp_to_i128(v: Fp) -> Option<i128> {
    let negative = fp_is_negative(v);
    let magnitude = if negative { -v } else { v };
    let bytes = magnitude.to_repr();
    let bytes = bytes.as_ref();
    if bytes[16..].iter().any(|b| *b != 0) {
        return None;
    }
    let m = u128::from_le_bytes(bytes[..16].try_into().unwrap());
    if m >= 1u128 << 126 {
        return None;
    }
    Some(if negative { -(m as i128) } else { m as i128 })
}

// ---------------------------------------------------------------------------
// Fixed-point quantization (operates on Fp)
// ---------------------------------------------------------------------------

/// Quantize a float to a fixed-point Fp field element.
pub fn quantize_to_fp(value: f64, precision_bits: u32) -> Fp {
    let q = crate::prove::field::quantize(value, precision_bits);
    if q >= 0 {
        Fp::from(q as u64)
    } else {
        -Fp::from((-q) as u64)
    }
}

/// Dequantize an Fp field element back to a float.
pub fn dequantize_from_fp(v: Fp, precision_bits: u32) -> f64 {
    let i = fp_to_i64(v);
    let scale = crate::prove::field::quantization_scale(precision_bits) as f64;
    i as f64 / scale
}

/// The quantization scale as an Fp field element.
pub fn scale_fp(precision_bits: u32) -> Fp {
    Fp::from(crate::prove::field::quantization_scale(precision_bits) as u64)
}

// ---------------------------------------------------------------------------
// Value ↔ Fp conversion
// ---------------------------------------------------------------------------

use crate::prove::error::ProvingError;
use crate::prove::types::Value;

/// Parse a 64-char hex string into an Fp field element.
pub fn hex_str_to_fp(s: &str) -> Result<Fp, ProvingError> {
    if s.len() != 64 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ProvingError::other(format!(
            "Invalid hex field element: '{}'",
            s
        )));
    }
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| ProvingError::other(format!("Invalid hex in '{}'", s)))?;
    }
    let repr = <Fp as PrimeField>::Repr::from(bytes);
    Option::from(Fp::from_repr(repr))
        .ok_or_else(|| ProvingError::other(format!("Hex '{}' is not a valid field element", s)))
}

/// Convert a `Value` to an Fp field element.
/// Integers → direct. Floats → quantized. Bools → 0/1.
pub fn value_to_fp(value: &Value, precision_bits: u32) -> Result<Fp, ProvingError> {
    match value {
        Value::Integer(v) => Ok(i64_to_fp(*v)),
        Value::Float(v) => Ok(quantize_to_fp(*v, precision_bits)),
        Value::Bool(v) => Ok(if *v { Fp::one() } else { Fp::zero() }),
        Value::Str(s) => {
            // Try hex-encoded field element bytes first (used for Poseidon hashes)
            if s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()) {
                if let Ok(fp) = hex_str_to_fp(s) {
                    return Ok(fp);
                }
            }
            // Fall back to parsing as integer
            let v: i64 = s.parse().map_err(|_| {
                ProvingError::other(format!("Cannot convert string '{}' to field element", s))
            })?;
            Ok(i64_to_fp(v))
        }
        Value::None => Ok(Fp::zero()),
        Value::List(_) => Err(ProvingError::other(
            "Cannot convert list to a single field element",
        )),
    }
}

/// Convert an Fp field element to a `Value`.
/// `is_float`: if true, dequantize to Float; otherwise, convert to Integer.
pub fn fp_to_value(v: Fp, is_float: bool, precision_bits: u32) -> Value {
    if is_float {
        Value::Float(dequantize_from_fp(v, precision_bits))
    } else {
        Value::Integer(fp_to_i64(v))
    }
}

// ---------------------------------------------------------------------------
// Pure computation functions (used by both mock and halo2 synthesizers)
// ---------------------------------------------------------------------------

/// Evaluate a polynomial via Horner's method, with **leading-first** coefficient
/// layout: `coefs = [a_{n-1}, a_{n-2}, ..., a_1, a_0]` so that the polynomial is
/// `p(x) = a_{n-1} * x^{n-1} + a_{n-2} * x^{n-2} + ... + a_1 * x + a_0`.
///
/// Recurrence: `acc_0 = coefs[0]`, `acc_{i+1} = x * acc_i + coefs[i+1]`.
/// All values are Fp field elements in fixed-point representation (scaled by 2^precision_bits).
///
/// NOTE: This matches the layout of `LOG2_COEFS`, `EXP2_COEFS` and `SIN_COEFS`
/// in this module — they are stored leading-first (`ATAN_COEFS` and
/// `ATAN_ODD_U_COEFS` are not). A previous
/// version of this function iterated in the opposite direction and treated
/// `coefs[0]` as the constant term, which silently produced wrong values for
/// the transcendentals (log(2) ≈ -11097 instead of 0.693, etc).
pub fn horner_eval(x: Fp, coefs: &[Fp], precision_bits: u32) -> Fp {
    assert!(!coefs.is_empty());
    let mut acc = coefs[0];
    for i in 1..coefs.len() {
        let (prod, _) = fp_mul_rescale(x, acc, precision_bits);
        acc = prod + coefs[i];
    }
    acc
}

/// Signed decomposition: returns (abs, is_negative_bit).
pub fn signed_decompose(v: Fp) -> (Fp, bool) {
    let is_neg = fp_is_negative(v);
    let abs = if is_neg { -v } else { v };
    (abs, is_neg)
}

/// Fixed-point multiplication with rescaling.
///
/// Computes `result = floor((a * b) / scale)` and `remainder = (a * b) - result * scale`,
/// where `scale = 2^precision_bits`. Uses i128 integer division (via `div_euclid` /
/// `rem_euclid`) — NOT field modular inverse. The field-element `scale.invert()`
/// approach is unsound: for non-trivial products it wraps the prime modulus and
/// produces a value unrelated to integer division.
///
/// `div_euclid` gives `0 <= remainder < scale.abs()`, matching the div_mod gate's
/// intended remainder-range semantics.
pub fn fp_mul_rescale(a: Fp, b: Fp, precision_bits: u32) -> (Fp, Fp) {
    let a_signed = fp_to_i64(a) as i128;
    let b_signed = fp_to_i64(b) as i128;
    let raw = a_signed * b_signed;
    fp_rescale_raw_i128(raw, precision_bits)
}

/// Rescale an already-accumulated raw fixed-point integer by `2^precision_bits`.
pub fn fp_rescale_raw_i128(raw: i128, precision_bits: u32) -> (Fp, Fp) {
    let scale = 1i128 << precision_bits;
    let quotient = raw.div_euclid(scale);
    let remainder = raw.rem_euclid(scale); // 0 <= remainder < scale
    let result_fp = i128_to_fp(quotient);
    let remainder_fp = Fp::from(remainder as u64); // remainder is non-negative and < 2^precision_bits
    (result_fp, remainder_fp)
}

/// Fused fixed-point inner product:
/// `floor(sum_i raw(a_i*b_i) / 2^precision_bits)`.
///
/// This matches halo2-graph's `FixedPointChip::inner_product` shape and is
/// closer to NumPy real arithmetic than adding already-rounded per-product
/// fixed-point quotients.
pub fn fp_product_sum_rescale(pairs: &[(Fp, Fp)], precision_bits: u32) -> (Fp, Fp) {
    let raw = pairs.iter().fold(0i128, |acc, (a, b)| {
        acc + (fp_to_i64(*a) as i128) * (fp_to_i64(*b) as i128)
    });
    fp_rescale_raw_i128(raw, precision_bits)
}

/// Fixed-point division with pre-scaling: `a * 2^P / b`, rounded by
/// [`Rounding::Euclid`] (remainder in `[0, |b|)`), i.e. toward negative
/// infinity for a positive divisor and toward positive infinity for a
/// negative one. Division by zero yields zero (see [`div_rem_or_zero`]).
pub fn fp_div_prescale(a: Fp, b: Fp, precision_bits: u32) -> Fp {
    let numerator = (fp_to_i64(a) as i128) << precision_bits;
    let (q, _) = div_rem_or_zero(numerator, fp_to_i64(b) as i128, Rounding::Euclid);
    i128_to_fp(q)
}

/// Encode a signed i128 as an Fp field element.
/// Mirrors the encoding used by `quantize` + `quantize_to_fp`: positive values
/// map to their natural Fp representation; negative values map to `-Fp::from(|v|)`.
pub fn i128_to_fp(v: i128) -> Fp {
    if v >= 0 {
        let u = v as u128;
        let lo = (u & 0xFFFF_FFFF_FFFF_FFFFu128) as u64;
        let hi = (u >> 64) as u64;
        // Fp::from(u64) for lo + Fp::from(u64) * 2^64 for hi.
        if hi == 0 {
            Fp::from(lo)
        } else {
            let two_pow_64 = Fp::from_u128(1u128 << 64);
            Fp::from(lo) + Fp::from(hi) * two_pow_64
        }
    } else {
        // -(2^127) is the only i128 whose negation overflows; handle via u128 cast.
        let u = (v as i128).unsigned_abs();
        let lo = (u & 0xFFFF_FFFF_FFFF_FFFFu128) as u64;
        let hi = (u >> 64) as u64;
        let pos = if hi == 0 {
            Fp::from(lo)
        } else {
            let two_pow_64 = Fp::from_u128(1u128 << 64);
            Fp::from(lo) + Fp::from(hi) * two_pow_64
        };
        -pos
    }
}

// ---------------------------------------------------------------------------
// Integer division conventions
// ---------------------------------------------------------------------------

/// How a signed integer division rounds its quotient. Each convention fixes
/// the sign of the remainder `r = a - b*q`; in all three `|r| < |b|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Toward negative infinity: `r` is zero or has the sign of `b`.
    /// Python's `//` and `%`, NumPy's `floor_divide` and `remainder`.
    Floor,
    /// Euclidean: `0 <= r < |b|`. Used by fixed-point `/` ([`fp_div_prescale`]).
    Euclid,
    /// Toward zero: `r` is zero or has the sign of `a`. Python's `int()`.
    Trunc,
}

/// `(q, r)` with `a == b*q + r` and `|r| < |b|`, rounded by `rounding`.
///
/// A zero divisor yields `(0, 0)`. This is the one division-by-zero
/// convention of the mock, preprocess and halo2-base backends. NumPy integer
/// arrays give the same result; NumPy floats give ±inf or nan, which fixed
/// point cannot represent. Callers keep `|a|` below 2^126 (every fixed-point
/// intermediate does), so the native operations cannot overflow.
pub fn div_rem_or_zero(a: i128, b: i128, rounding: Rounding) -> (i128, i128) {
    if b == 0 {
        return (0, 0);
    }
    match rounding {
        Rounding::Euclid => (a.div_euclid(b), a.rem_euclid(b)),
        Rounding::Trunc => (a / b, a % b),
        Rounding::Floor => {
            let (q, r) = (a / b, a % b);
            if r != 0 && ((r < 0) != (b < 0)) {
                (q - 1, r + b)
            } else {
                (q, r)
            }
        }
    }
}

/// Float→int cast (truncation toward zero, Python's `int()`).
///
/// Interprets `v` as a fixed-point value scaled by `2^precision_bits` and
/// returns `q = trunc(v / 2^P)` with the remainder `r = v - q*2^P`, which has
/// the sign of `v` (or is zero) and `|r| < 2^P`.
pub fn fp_int_cast(v: Fp, precision_bits: u32) -> (Fp, Fp) {
    let (q, r) = div_rem_or_zero(fp_to_i64(v) as i128, 1i128 << precision_bits, Rounding::Trunc);
    (i128_to_fp(q), i128_to_fp(r))
}

/// Integer floor division and remainder with Python semantics
/// ([`Rounding::Floor`]): `-7 // 2 == -4`, `-7 % 2 == 1`, `7 % -2 == -1`.
/// A zero divisor yields `(0, 0)` (see [`div_rem_or_zero`]).
pub fn fp_floor_div(a: Fp, b: Fp) -> (Fp, Fp) {
    let (q, r) = div_rem_or_zero(fp_to_i64(a) as i128, fp_to_i64(b) as i128, Rounding::Floor);
    (i128_to_fp(q), i128_to_fp(r))
}

/// Float floor division `a // b` on Q-encoded values: `floor(a / b)`,
/// re-encoded as a Q value.
pub fn fp_float_floor_div(a: Fp, b: Fp, precision_bits: u32) -> Fp {
    fp_floor_div(a, b).0 * scale_fp(precision_bits)
}

/// Float remainder `a % b` on Q-encoded values. It has the divisor's sign,
/// as in Python and NumPy.
pub fn fp_float_mod(a: Fp, b: Fp) -> Fp {
    fp_floor_div(a, b).1
}

// ---------------------------------------------------------------------------
// Fixed-point square root and trigonometry
//
// These are the exact integer algorithms the halo2-base circuit constrains
// (`prove/halo2_base/synthesizer.rs`), so the mock backend and the circuit
// compute bit-identical values. A Q value whose magnitude is 2^64 or more is
// outside the fixed-point window: the circuit is unsatisfiable on it and
// these functions return 0.
// ---------------------------------------------------------------------------

/// Integer square root: the largest `y` with `y * y <= n`, for `n < 2^127`.
pub fn isqrt_u128(n: u128) -> u128 {
    let mut y = (n as f64).sqrt() as u128;
    while y * y > n {
        y -= 1;
    }
    while (y + 1) * (y + 1) <= n {
        y += 1;
    }
    y
}

/// `sqrt` on Q-encoded values, rounded down: the largest `y` with
/// `y^2 <= x * 2^P`. A negative input yields 0 (NumPy gives nan, which fixed
/// point cannot represent; 0 is the shared backend convention).
pub fn fp_sqrt(x: Fp, precision_bits: u32) -> Fp {
    match fp_to_i128(x) {
        Some(v) if v > 0 && v < (1i128 << 64) => {
            i128_to_fp(isqrt_u128((v as u128) << precision_bits) as i128)
        }
        _ => Fp::zero(),
    }
}

/// Bits of the admitted radicands of [`fp_sqrt_product_int`]: `n < 2^(126 - 2P)`,
/// the largest range whose roots `floor(sqrt(n) * 2^P)` stay below `2^63`, in
/// the fixed-point window (`2^62` at `P = 32`).
pub fn sqrt_int_bits(precision_bits: u32) -> u32 {
    126 - 2 * precision_bits
}

/// `sqrt(b * c)` of two integers as a Q value, rounded down:
/// `floor(sqrt(n) * 2^P)` for the field product `n = b * c`, the largest `y`
/// with `y^2 <= n * 2^(2P)` (`IR::SqrtProductIF`). Admits `0 <= b < 2^64` and
/// `0 <= n < 2^sqrt_int_bits(P)`, radicands far beyond the window of
/// `fp_sqrt(float(n))` (which needs `n < 2^31`); `None` outside, where every
/// backend has no proof. The bound on `b` lets a caller show that the field
/// product is the integer product (`|c| < p / 2^65` suffices).
pub fn fp_sqrt_product_int(b: Fp, c: Fp, precision_bits: u32) -> Option<Fp> {
    fp_to_i128(b).filter(|v| (0..1i128 << 64).contains(v))?;
    let n = fp_to_i128(b * c).filter(|v| (0..1i128 << sqrt_int_bits(precision_bits)).contains(v))?;
    Some(i128_to_fp(isqrt_u128((n as u128) << (2 * precision_bits)) as i128))
}

/// π and π/2 in Q-encoding (rounded). Sine reduces its argument modulo
/// `2 * pi_q` and folds it onto `[0, half_pi_q]`.
pub fn pi_q(precision_bits: u32) -> i128 {
    crate::prove::field::quantize(std::f64::consts::PI, precision_bits)
}

pub fn half_pi_q(precision_bits: u32) -> i128 {
    crate::prove::field::quantize(std::f64::consts::FRAC_PI_2, precision_bits)
}


/// [`SIN_ODD_U_COEFS`] Q-encoded and leading-first, for [`horner_eval`].
pub fn sin_odd_coefs_q(precision_bits: u32) -> Vec<Fp> {
    SIN_ODD_U_COEFS.iter().rev().map(|c| quantize_to_fp(*c, precision_bits)).collect()
}

/// [`ATAN_ODD_U_COEFS`] Q-encoded and leading-first, for [`horner_eval`].
pub fn atan_odd_coefs_q(precision_bits: u32) -> Vec<Fp> {
    ATAN_ODD_U_COEFS.iter().rev().map(|c| quantize_to_fp(*c, precision_bits)).collect()
}

/// `sin` on Q-encoded values, by the circuit's algorithm:
///
/// 1. `m = |x|`; `r = m mod (2 * pi_q)`;
/// 2. fold `r >= pi_q` onto `r - pi_q` (sin(r) = -sin(r - π));
/// 3. mirror `r > half_pi_q` onto `pi_q - r` (sin(r) = sin(π - r)), so
///    `s = r` lies in `[0, π/2]`;
/// 4. `s * P(s^2)` with [`SIN_ODD_U_COEFS`] by [`horner_eval`] (every
///    product rounded down, as `mul_f` does);
/// 5. negate when exactly one of "x < 0" and "folded" holds.
///
/// Measured maximum error against libm: 9.0e-9 on `[-20, 20]`, 4e-8 on
/// `[-1000, 1000]` (the range reduction loses about one Q unit per period).
pub fn fp_sin(x: Fp, precision_bits: u32) -> Fp {
    let Some(v) = fp_to_i128(x) else {
        return Fp::zero();
    };
    let m = v.unsigned_abs();
    if m >= 1u128 << 64 {
        return Fp::zero();
    }
    let pi = pi_q(precision_bits) as u128;
    let half_pi = half_pi_q(precision_bits) as u128;
    let r = m % (2 * pi);
    let (r, folded) = if r >= pi { (r - pi, true) } else { (r, false) };
    let s = if r > half_pi { pi - r } else { r };
    let s = i128_to_fp(s as i128);
    let (u, _) = fp_mul_rescale(s, s, precision_bits);
    let odd = horner_eval(u, &sin_odd_coefs_q(precision_bits), precision_bits);
    let p = fp_mul_rescale(s, odd, precision_bits).0;
    if (v < 0) != folded {
        -p
    } else {
        p
    }
}

/// `cos(x) = sin(x + π/2)` on Q-encoded values (the circuit's algorithm).
pub fn fp_cos(x: Fp, precision_bits: u32) -> Fp {
    fp_sin(x + i128_to_fp(half_pi_q(precision_bits)), precision_bits)
}

/// `arcsin` on Q-encoded values, by the circuit's algorithm:
/// `arcsin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))`. The square root is the
/// exact `floor(sqrt(2^(2P) - x_q^2))` of the raw value, so no rounding of
/// `x^2` enters it; the atan argument lies in `[-1, 1]`; the quotient is
/// [`fp_div_prescale`] and the polynomial [`ATAN_ODD_U_COEFS`]. Outside
/// `[-1, 1]` (where NumPy gives nan) the result is 0.
///
/// Measured maximum error against libm on `[-1, 1]`: 5.9e-9 (largest
/// near `|x| = 1`, where `arcsin` is ill-conditioned).
pub fn fp_arcsin(x: Fp, precision_bits: u32) -> Fp {
    let one = 1i128 << precision_bits;
    let v = match fp_to_i128(x) {
        Some(v) if v.abs() <= one => v,
        _ => 0,
    };
    let root = isqrt_u128((one * one - v * v) as u128) as i128;
    let t = fp_div_prescale(i128_to_fp(v), i128_to_fp(one + root), precision_bits);
    let (u, _) = fp_mul_rescale(t, t, precision_bits);
    let p = horner_eval(u, &atan_odd_coefs_q(precision_bits), precision_bits);
    let (half, _) = fp_mul_rescale(t, p, precision_bits);
    half + half
}

/// `1 / ln 2` Q-encoded: `round(2^P / ln 2)` (6196328019 at `P = 32`).
pub fn inv_ln2_q(precision_bits: u32) -> i128 {
    crate::prove::field::quantize(std::f64::consts::LOG2_E, precision_bits)
}

/// [`EXP2_COEFS`] Q-encoded (already leading-first), for [`horner_eval`].
pub fn exp2_coefs_q(precision_bits: u32) -> Vec<Fp> {
    EXP2_COEFS.iter().map(|c| quantize_to_fp(*c, precision_bits)).collect()
}

/// The scale shift `S` of [`fp_exp`]'s last step, which the circuits
/// compute as `floor(t * 2^(i + S) / 2^S)` (= `floor(t * 2^i)`) for
/// `0 <= i + S < 128`, and as 0 for `i < -S`. Any `S >= P + 2` gives the
/// same values (`t < 2^(P+2)`, so `floor(t * 2^i) = 0` for `i <= -(P + 2)`);
/// 48 also keeps `t * 2^(i + S)` below `2^126` for every result inside the
/// window (`i <= 30`), the range `fp_to_i128` reads.
pub const EXP_SHIFT: i128 = 48;

/// `exp` on Q-encoded values, by the circuits' algorithm (halo2-base
/// `exp_f`, Noir `qexp`), so every backend computes the same value:
///
/// 1. `y = floor(x * inv_ln2_q / 2^P)` (a `mul_f`; `y` must lie in the
///    signed 64-bit window), so `exp(x) ~ 2^(y / 2^P)`;
/// 2. `i = floor(y / 2^P)`, `f = y mod 2^P`, so `y / 2^P = i + f / 2^P`
///    with `f / 2^P` in `[0, 1)`;
/// 3. `t = 2^(f / 2^P)` in `[1, 2]`, Q-encoded, by [`EXP2_COEFS`] with
///    [`horner_eval`] (every product rounded down, as `mul_f`);
/// 4. `t * 2^i`, exact for `i >= 0`, rounded down for `i < 0` (0 for
///    `i <= -(P + 2)`: underflow).
///
/// `None` (no proof on any backend) when `x` or `y` leaves the signed
/// 64-bit window or the result does (`exp(x) >= 2^(63 - P)`, i.e.
/// `x > 21.48` at `P = 32`).
///
/// Measured maximum error against libm on `[-40, 21.4]` at `P = 32`:
/// relative 9.4e-10 for results of at least 1, absolute 9.4e-10 (about 4 Q
/// units) for results below 1 (asserted by `fixed_point_exp_is_accurate`).
pub fn fp_exp(x: Fp, precision_bits: u32) -> Option<Fp> {
    let window = -(1i128 << 63)..(1i128 << 63);
    let scale = 1i128 << precision_bits;
    let v = fp_to_i128(x).filter(|v| window.contains(v))?;
    let y = (v * inv_ln2_q(precision_bits)).div_euclid(scale);
    if !window.contains(&y) {
        return None;
    }
    let (i, f) = (y.div_euclid(scale), y.rem_euclid(scale));
    let t = fp_to_i128(horner_eval(i128_to_fp(f), &exp2_coefs_q(precision_bits), precision_bits))?;
    let value = match i {
        i if i < -EXP_SHIFT => 0,
        i if i < 0 => t >> -i,
        i if i < EXP_SHIFT => t.checked_shl(i as u32).filter(|r| *r >> i == t && window.contains(r))?,
        _ => return None,
    };
    Some(i128_to_fp(value))
}

/// Compute ln(x) using native f64 math on the dequantized value.
pub fn fp_log(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.ln(), precision_bits)
}

/// Compute sinh: (exp(x) - exp(-x)) / 2 (`None` where an `exp` is outside
/// its domain, see [`fp_exp`]).
pub fn fp_sinh(x: Fp, precision_bits: u32) -> Option<Fp> {
    let ex = fp_exp(x, precision_bits)?;
    let enx = fp_exp(-x, precision_bits)?;
    let two = quantize_to_fp(2.0, precision_bits);
    Some(fp_div_prescale(ex - enx, two, precision_bits))
}

/// Compute cosh: (exp(x) + exp(-x)) / 2 (`None` where an `exp` is outside
/// its domain, see [`fp_exp`]).
pub fn fp_cosh(x: Fp, precision_bits: u32) -> Option<Fp> {
    let ex = fp_exp(x, precision_bits)?;
    let enx = fp_exp(-x, precision_bits)?;
    let two = quantize_to_fp(2.0, precision_bits);
    Some(fp_div_prescale(ex + enx, two, precision_bits))
}

/// Compute tan: sin(x) / cos(x).
pub fn fp_tan(x: Fp, precision_bits: u32) -> Fp {
    let s = fp_sin(x, precision_bits);
    let c = fp_cos(x, precision_bits);
    fp_div_prescale(s, c, precision_bits)
}

/// Compute tanh: sinh(x) / cosh(x) (`None` where an `exp` is outside its
/// domain, see [`fp_exp`]).
pub fn fp_tanh(x: Fp, precision_bits: u32) -> Option<Fp> {
    let s = fp_sinh(x, precision_bits)?;
    let c = fp_cosh(x, precision_bits)?;
    Some(fp_div_prescale(s, c, precision_bits))
}

/// Compute atan2(y, x) using native f64 math on the dequantized values.
pub fn fp_atan2(y: Fp, x: Fp, precision_bits: u32) -> Fp {
    let yv = dequantize_from_fp(y, precision_bits);
    let xv = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(yv.atan2(xv), precision_bits)
}

/// Compute arccos(x) using native f64 math on the dequantized value.
pub fn fp_arccos(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.acos(), precision_bits)
}

// ---------------------------------------------------------------------------
// Prover hints (IR::Hint)
// ---------------------------------------------------------------------------

/// The total order of every sort hint: the signed reading of a field
/// element (the integer of smallest magnitude it represents, the reading of
/// [`fp_is_negative`]). On the signed 64-bit window, where Zinnia's int64
/// values and Q32 floats live, it is their integer order, the order the
/// circuits' comparisons decide. `noir_fpc::signed_lt` implements the same
/// order for the Noir backend's hints.
pub fn signed_cmp(a: Fp, b: Fp) -> std::cmp::Ordering {
    let (a_negative, b_negative) = (fp_is_negative(a), fp_is_negative(b));
    if a_negative != b_negative {
        // The negative one is smaller.
        return b_negative.cmp(&a_negative);
    }
    // Same sign: the readings `v` and `v - p` keep the order of the field
    // representatives, compared as integers from the most significant byte.
    let (a_repr, b_repr) = (a.to_repr(), b.to_repr());
    a_repr.as_ref().iter().rev().cmp(b_repr.as_ref().iter().rev())
}

/// The stable ascending argsort of `values` under [`signed_cmp`]: entry `r`
/// is the original position of the element of rank `r`, equal elements in
/// their original order (NumPy's `kind="stable"`).
pub fn stable_argsort(values: &[Fp]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    // `sort_by` is a stable sort.
    order.sort_by(|&i, &j| signed_cmp(values[i], values[j]));
    order
}

/// Every lane of the hint `kind` applied to its operands (see
/// [`crate::ir_defs::HintKind`]).
pub fn hint_values(kind: crate::ir_defs::HintKind, operands: &[Fp]) -> Vec<Fp> {
    use crate::ir_defs::HintKind;
    match kind {
        HintKind::StableArgsort => {
            stable_argsort(operands).into_iter().map(|i| Fp::from(i as u64)).collect()
        }
        HintKind::SortedValues => stable_argsort(operands).into_iter().map(|i| operands[i]).collect(),
        HintKind::CountAtMost => Vec::new(), // not lanes; see `hint_lane`
        HintKind::MaskedSortedValues => {
            let (values, active) = operands.split_at(operands.len() / 2);
            let active: Vec<Fp> = values
                .iter()
                .zip(active)
                .filter(|(_, flag)| **flag != Fp::zero())
                .map(|(v, _)| *v)
                .collect();
            let mut lanes: Vec<Fp> = stable_argsort(&active).into_iter().map(|i| active[i]).collect();
            lanes.resize(values.len(), Fp::zero());
            lanes
        }
        HintKind::PositionOneHot => position_one_hot(operands),
        HintKind::Negative => {
            let negative = operands.first().is_some_and(|x| fp_is_negative(*x));
            vec![if negative { Fp::one() } else { Fp::zero() }]
        }
    }
}

/// The lanes of `HintKind::PositionOneHot` for the operands `[x, n]`: a
/// one-hot of NumPy's position of `x` on an axis of length `n` (`x` or
/// `x + n`, the end `n` included), all zero when `x` names no position. No
/// lanes when `n` is not a small non-negative integer (only a malformed
/// IR has such an `n`; every lane then reads as 0).
fn position_one_hot(operands: &[Fp]) -> Vec<Fp> {
    const MAX_AXIS: i128 = 1 << 24;
    let (Some(x), Some(n)) = (operands.first(), operands.get(1)) else { return Vec::new() };
    let Some(n) = fp_to_i128(*n).filter(|n| (0..MAX_AXIS).contains(n)) else { return Vec::new() };
    let position = fp_to_i128(*x).map(|x| if x < 0 { x + n } else { x }).filter(|p| (0..=n).contains(p));
    (0..=n).map(|r| if Some(r) == position { Fp::one() } else { Fp::zero() }).collect()
}

/// The decomposition `K = q * 2^index_bits + i` of a packed sort key
/// (`IR::PackKey`) with `i` in `[0, 2^index_bits)`: `(q, i)`, the value and
/// the index the key was packed from. A field element with no signed
/// reading (never a packed key) gives `(0, 0)`.
pub fn packed_key_parts(key: Fp, index_bits: u32) -> (Fp, Fp) {
    let scale = 1i128 << index_bits;
    match fp_to_i128(key) {
        Some(k) => (i128_to_fp(k.div_euclid(scale)), i128_to_fp(k.rem_euclid(scale))),
        None => (Fp::zero(), Fp::zero()),
    }
}

/// Lane `rank` of the hint `kind` applied to `operands`, the value of an
/// `IR::Hint` statement on every backend; 0 for a rank outside the lanes.
/// The circuits never trust it: each use of a hint pins it with
/// constraints that admit only this value.
/// `HintKind::CountAtMost` takes the rank operand as its query instead.
pub fn hint_lane(kind: crate::ir_defs::HintKind, operands: &[Fp], rank: Fp) -> Fp {
    if kind == crate::ir_defs::HintKind::CountAtMost {
        let count = operands.iter().filter(|&&a| signed_cmp(a, rank) != std::cmp::Ordering::Greater).count();
        return Fp::from(count as u64);
    }
    let lanes = hint_values(kind, operands);
    fp_to_i128(rank)
        .and_then(|r| usize::try_from(r).ok())
        .and_then(|r| lanes.get(r).copied())
        .unwrap_or(Fp::zero())
}

/// Simplified Poseidon permutation (4 rounds, t=3).
/// Returns the first state element as the hash output.
pub fn fp_poseidon(inputs: &[Fp]) -> Fp {
    let mut state = [Fp::zero(), Fp::zero(), Fp::zero()];
    for (i, input) in inputs.iter().enumerate() {
        let idx = i % 2;
        state[idx] = state[idx] + input;
        if idx == 1 || i == inputs.len() - 1 {
            for _round in 0..4 {
                // S-box: x^5
                for j in 0..3 {
                    let x2 = state[j] * state[j];
                    let x4 = x2 * x2;
                    state[j] = x4 * state[j];
                }
                // Linear mix
                let s0 = state[0] + state[1];
                let s1 = state[1] + state[2];
                let s2 = state[2] + state[0];
                state = [s0, s1, s2];
            }
        }
    }
    state[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i64_fp_round_trip() {
        for v in [-100i64, -1, 0, 1, 42, 1000] {
            assert_eq!(fp_to_i64(i64_to_fp(v)), v, "Round-trip failed for {}", v);
        }
    }

    #[test]
    fn test_fp_is_negative() {
        assert!(!fp_is_negative(Fp::zero()));
        assert!(!fp_is_negative(Fp::from(1)));
        assert!(fp_is_negative(-Fp::from(1)));
        assert!(fp_is_negative(-Fp::from(100)));
    }

    #[test]
    fn test_quantize_dequantize_fp() {
        let prec = 32;
        for v in [0.0, 1.0, -1.0, 3.14, -2.71] {
            let q = quantize_to_fp(v, prec);
            let d = dequantize_from_fp(q, prec);
            assert!(
                (d - v).abs() < 1e-6,
                "Round-trip failed for {}: got {}",
                v,
                d
            );
        }
    }

    #[test]
    fn test_horner_eval_constant() {
        // p(x) = 5 — precision_bits=0 for raw (unscaled) arithmetic
        let result = horner_eval(Fp::from(10), &[Fp::from(5)], 0);
        assert_eq!(result, Fp::from(5));
    }

    #[test]
    fn test_horner_eval_linear() {
        // Leading-first layout: coefs = [a_1, a_0] for p(x) = a_1*x + a_0.
        // p(x) = 2*x + 3 at x=5 → 13.
        let result = horner_eval(Fp::from(5), &[Fp::from(2), Fp::from(3)], 0);
        assert_eq!(result, Fp::from(13));
    }

    #[test]
    fn test_horner_eval_log2_in_domain() {
        // The `LOG2_COEFS` polynomial is fit on [2, 4) — verify the kernel-level
        // Horner produces the expected log2 values when invoked on inputs in that
        // domain. This is a regression guard for the leading-first ordering.
        //
        // Tolerance: 1e-2 reflects accumulated Q32 truncation across the 14-step
        // Horner chain. The leading coefficient of `LOG2_COEFS` is ~3e-8, which
        // quantizes to ~7 bits in Q32 — single-step truncation produces ~5e-3
        // error at the upper end of [2, 4). For a tighter bound, internal
        // arithmetic would need to widen to Q48 or Q64.
        let precision_bits: u32 = 32;
        let log2_coefs: Vec<Fp> = LOG2_COEFS
            .iter()
            .map(|c| quantize_to_fp(*c, precision_bits))
            .collect();
        let cases: &[(f64, f64)] = &[
            (2.0, 1.0),
            (2.5, (2.5f64).log2()),
            (3.0, (3.0f64).log2()),
            (3.5, (3.5f64).log2()),
        ];
        for &(x, expected) in cases {
            let xq = quantize_to_fp(x, precision_bits);
            let result = horner_eval(xq, &log2_coefs, precision_bits);
            let got = dequantize_from_fp(result, precision_bits);
            assert!(
                (got - expected).abs() < 1e-2,
                "horner_eval(log2 polynomial, x={x}) = {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn fixed_point_exp_values_match_noir_qexp() {
        // The same values are asserted for Noir's `qexp` in
        // benchmarking/noir_fpc (test_qexp_matches_the_kernel), and the
        // constants are noir_fpc's INV_LN2_Q and EXP2_COEFS_Q.
        let q = 1i128 << 32;
        assert_eq!(inv_ln2_q(32), 6196328019);
        let coefs: Vec<i128> = exp2_coefs_q(32).into_iter().map(|c| fp_to_i128(c).unwrap()).collect();
        assert_eq!(coefs, vec![
            0, 2, 31, 437, 5676, 65510, 661577, 5726720, 41309550, 238388332, 1031764991,
            2977044472, 4294967296,
        ]);
        let exp = |x: i128| fp_exp(i128_to_fp(x), 32).map(|v| fp_to_i128(v).unwrap());
        assert_eq!(exp(0), Some(q));
        assert_eq!(exp(q), Some(11674931554));
        assert_eq!(exp(-q), Some(1580030168));
        assert_eq!(exp(-20 * q), Some(8));
        assert_eq!(exp(-30 * q), Some(0));
        assert_eq!(exp(-100 * q), Some(0));
        assert_eq!(exp(-(1i128 << 62)), Some(0));
        assert_eq!(exp(21 * q), Some(5664270453422686208));
        assert_eq!(exp(22 * q), None);
        assert_eq!(exp(1i128 << 62), None);
    }

    #[test]
    fn fixed_point_exp_is_accurate() {
        let (mut worst_relative, mut worst_absolute) = (0f64, 0f64);
        for step in 0..=61_400 {
            let x = -40.0 + step as f64 * 1e-3;
            let xq = quantize_to_fp(x, 32);
            let exact = dequantize_from_fp(xq, 32).exp();
            let got = dequantize_from_fp(fp_exp(xq, 32).expect("inside the domain"), 32);
            if exact >= 1.0 {
                worst_relative = worst_relative.max((got - exact).abs() / exact);
            } else {
                worst_absolute = worst_absolute.max((got - exact).abs());
            }
        }
        assert!(worst_relative < 1e-9, "relative error {worst_relative}");
        assert!(worst_absolute < 5.0 / 4294967296.0, "absolute error {worst_absolute}");
    }

    #[test]
    fn test_horner_eval_exp2_in_domain() {
        // `EXP2_COEFS` is fit on [0, 1). Coefficient magnitudes are large enough
        // that Q32 quantization preserves precision well; this test asserts that.
        let precision_bits: u32 = 32;
        let exp2_coefs: Vec<Fp> = EXP2_COEFS
            .iter()
            .map(|c| quantize_to_fp(*c, precision_bits))
            .collect();
        let cases: &[f64] = &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9];
        for &x in cases {
            let xq = quantize_to_fp(x, precision_bits);
            let result = horner_eval(xq, &exp2_coefs, precision_bits);
            let got = dequantize_from_fp(result, precision_bits);
            let expected = 2.0f64.powf(x);
            assert!(
                (got - expected).abs() < 1e-4,
                "horner_eval(exp2 polynomial, x={x}) = {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn test_i128_to_fp_round_trip() {
        // Positive, negative, large magnitudes
        let cases: &[i128] = &[
            0,
            1,
            -1,
            42,
            -42,
            (1i128 << 32) - 1,
            -(1i128 << 32) + 1,
            (1i128 << 62),
            -(1i128 << 62),
            (1i128 << 64),
            -(1i128 << 64),
            (1i128 << 100),
            -(1i128 << 100),
        ];
        for &v in cases {
            let fp = i128_to_fp(v);
            // For values that fit in i64, fp_to_i64 should round-trip.
            if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
                assert_eq!(
                    fp_to_i64(fp),
                    v as i64,
                    "i64-range round-trip failed for {}",
                    v
                );
            }
            // Sign check
            if v < 0 {
                assert!(fp_is_negative(fp), "expected negative for {}", v);
            } else if v > 0 {
                assert!(!fp_is_negative(fp), "expected non-negative for {}", v);
            }
        }
    }

    #[test]
    fn fp_mul_rescale_matches_integer_division() {
        let precision_bits: u32 = 32;
        let scale = (1i128 << precision_bits) as f64;
        let pairs: &[(f64, f64)] = &[
            (1.066508022690349, 1.066508022690349), // the fuzz repro
            (2.0, 3.0),
            (-1.5, 2.5),
            (0.1, 0.1),
            (1234.5, 0.001),
            (-2.5264763958797793, 2.5264763958797793),
            (0.0, 5.0),
            (5.0, 0.0),
        ];
        for &(a, b) in pairs {
            let aq = quantize_to_fp(a, precision_bits);
            let bq = quantize_to_fp(b, precision_bits);
            let (result, _rem) = fp_mul_rescale(aq, bq, precision_bits);
            let got = dequantize_from_fp(result, precision_bits);
            let expected = a * b;
            let err = (got - expected).abs();
            let one_ulp = 1.0 / scale;
            // Tolerance: propagated quantization error is bounded by
            // |b|*ulp(a) + |a|*ulp(b) + 1*ulp(result) plus a small safety factor.
            let tol = (a.abs() + b.abs() + 4.0) * one_ulp;
            assert!(
                err <= tol,
                "fp_mul_rescale({a}, {b}) -> {got}; expected {expected}; err {err:.2e}, tol {tol:.2e}, ULP {one_ulp:.2e}",
            );
        }
    }

    #[test]
    fn fp_mul_rescale_remainder_in_range() {
        let precision_bits: u32 = 32;
        let scale = 1i128 << precision_bits;
        let pairs: &[(f64, f64)] = &[
            (1.066508022690349, 1.066508022690349),
            (2.0, 3.0),
            (-1.5, 2.5),
            (0.1, 0.1),
            (1234.5, 0.001),
            (-2.5264763958797793, 2.5264763958797793),
            (-7.25, -3.5),
            (1e-3, 1e-3),
        ];
        for &(a, b) in pairs {
            let aq = quantize_to_fp(a, precision_bits);
            let bq = quantize_to_fp(b, precision_bits);
            let (_q, rem) = fp_mul_rescale(aq, bq, precision_bits);
            // remainder should round-trip as a non-negative integer in [0, scale).
            let rem_i64 = fp_to_i64(rem);
            assert!(
                rem_i64 >= 0,
                "fp_mul_rescale({a}, {b}) remainder {} is negative",
                rem_i64,
            );
            let rem_i128 = rem_i64 as i128;
            assert!(
                rem_i128 < scale,
                "fp_mul_rescale({a}, {b}) remainder {} >= scale {}",
                rem_i128,
                scale,
            );
        }
    }

    #[test]
    fn fp_mul_rescale_div_mod_identity() {
        // The div_mod gate enforces raw == scale * result + rem in Fp.
        // Verify this identity holds for our kernel output.
        let precision_bits: u32 = 32;
        let scale_fp_val = scale_fp(precision_bits);
        let pairs: &[(f64, f64)] = &[
            (1.066508022690349, 1.066508022690349),
            (-1.5, 2.5),
            (1234.5, 0.001),
            (-2.5264763958797793, 2.5264763958797793),
        ];
        for &(a, b) in pairs {
            let aq = quantize_to_fp(a, precision_bits);
            let bq = quantize_to_fp(b, precision_bits);
            let raw = aq * bq;
            let (result, rem) = fp_mul_rescale(aq, bq, precision_bits);
            let recomputed = scale_fp_val * result + rem;
            assert_eq!(recomputed, raw, "div_mod identity failed for ({a}, {b})",);
        }
    }

    #[test]
    fn fp_div_prescale_matches_integer_division() {
        let precision_bits: u32 = 32;
        let scale = (1i128 << precision_bits) as f64;
        let pairs: &[(f64, f64)] = &[
            (1.0, 1.0),
            (2.0, 3.0),
            (-1.5, 2.5),
            (1234.5, 0.001),
            (1.0, 1.066508022690349),
            (10.0, -4.0),
        ];
        for &(a, b) in pairs {
            let aq = quantize_to_fp(a, precision_bits);
            let bq = quantize_to_fp(b, precision_bits);
            let result = fp_div_prescale(aq, bq, precision_bits);
            let got = dequantize_from_fp(result, precision_bits);
            let expected = a / b;
            let err = (got - expected).abs();
            // Error is dominated by quantization of `b`: q(b) carries up to 0.5 ULP
            // absolute, which becomes |a|/b^2 * 0.5 * ulp in the quotient.
            let one_ulp = 1.0 / scale;
            let propagated = (a.abs() / (b * b).abs()) * one_ulp + 4.0 * one_ulp;
            let tol = propagated.max(4.0 * one_ulp);
            assert!(
                err <= tol,
                "fp_div_prescale({a}, {b}) -> {got}; expected {expected}; err {err:.2e}, tol {tol:.2e}",
            );
        }
    }

    #[test]
    fn division_conventions() {
        // (a, b) -> (floor, euclid, trunc) quotient/remainder pairs.
        let cases = [
            (7, 2, (3, 1), (3, 1), (3, 1)),
            (-7, 2, (-4, 1), (-4, 1), (-3, -1)),
            (7, -2, (-4, -1), (-3, 1), (-3, 1)),
            (-7, -2, (3, -1), (4, 1), (3, -1)),
            (6, -3, (-2, 0), (-2, 0), (-2, 0)),
        ];
        for (a, b, floor, euclid, trunc) in cases {
            assert_eq!(div_rem_or_zero(a, b, Rounding::Floor), floor, "{a} floor {b}");
            assert_eq!(div_rem_or_zero(a, b, Rounding::Euclid), euclid, "{a} euclid {b}");
            assert_eq!(div_rem_or_zero(a, b, Rounding::Trunc), trunc, "{a} trunc {b}");
        }
        for rounding in [Rounding::Floor, Rounding::Euclid, Rounding::Trunc] {
            assert_eq!(div_rem_or_zero(5, 0, rounding), (0, 0));
        }
    }

    #[test]
    fn integer_floor_division_follows_python() {
        let q = |a: i64, b: i64| fp_to_i64(fp_floor_div(i64_to_fp(a), i64_to_fp(b)).0);
        let r = |a: i64, b: i64| fp_to_i64(fp_floor_div(i64_to_fp(a), i64_to_fp(b)).1);
        assert_eq!((q(-7, 2), r(-7, 2)), (-4, 1));
        assert_eq!((q(7, -2), r(7, -2)), (-4, -1));
        assert_eq!((q(-7, -2), r(-7, -2)), (3, -1));
        assert_eq!((q(5, 0), r(5, 0)), (0, 0));
        let prec = 32;
        let mod_f = |a: f64, b: f64| dequantize_from_fp(fp_float_mod(quantize_to_fp(a, prec), quantize_to_fp(b, prec)), prec);
        let fdiv_f = |a: f64, b: f64| {
            dequantize_from_fp(fp_float_floor_div(quantize_to_fp(a, prec), quantize_to_fp(b, prec), prec), prec)
        };
        assert_eq!((fdiv_f(7.5, -2.0), mod_f(7.5, -2.0)), (-4.0, -0.5));
        assert_eq!((fdiv_f(-7.5, 2.0), mod_f(-7.5, 2.0)), (-4.0, 0.5));
    }

    #[test]
    fn fp_to_i128_round_trips_and_rejects_large_values() {
        for v in [0i128, 1, -1, 1 << 95, -(1 << 95), (1 << 126) - 1] {
            assert_eq!(fp_to_i128(i128_to_fp(v)), Some(v));
        }
        assert_eq!(fp_to_i128(i128_to_fp(1 << 126)), None);
        assert_eq!(fp_to_i128(i128_to_fp(1 << 126) * Fp::from(4u64)), None);
    }

    #[test]
    fn fp_sqrt_is_the_exact_floor_root() {
        let prec = 32;
        for x in [0.0, 1e-9, 0.25, 2.0, 3.0, 1e6, 2e9] {
            let v = fp_to_i128(quantize_to_fp(x, prec)).unwrap() as u128;
            let y = fp_to_i128(fp_sqrt(quantize_to_fp(x, prec), prec)).unwrap() as u128;
            let n = v << prec;
            assert!(y * y <= n && n < (y + 1) * (y + 1), "sqrt({x})");
        }
        assert_eq!(fp_sqrt(quantize_to_fp(-4.0, prec), prec), Fp::zero());
    }

    /// Maximum error against libm, measured on a dense grid: the bounds
    /// stated in the doc comments of `fp_sin` and `fp_arcsin`.
    #[test]
    fn fixed_point_trigonometry_is_accurate() {
        let prec = 32;
        let error = |got: Fp, want: f64| (dequantize_from_fp(got, prec) - want).abs();
        let grid = |lo: f64, hi: f64, n: usize| (0..=n).map(move |i| lo + (hi - lo) * i as f64 / n as f64);
        let mut sin_cos: f64 = 0.0;
        for x in grid(-20.0, 20.0, 20_000) {
            let xq = quantize_to_fp(x, prec);
            let exact = dequantize_from_fp(xq, prec);
            sin_cos = sin_cos.max(error(fp_sin(xq, prec), exact.sin()));
            sin_cos = sin_cos.max(error(fp_cos(xq, prec), exact.cos()));
        }
        assert!(sin_cos < 1e-8, "sin/cos error {sin_cos}");
        let mut far: f64 = 0.0;
        for x in grid(-1000.0, 1000.0, 20_000) {
            let xq = quantize_to_fp(x, prec);
            far = far.max(error(fp_sin(xq, prec), dequantize_from_fp(xq, prec).sin()));
        }
        assert!(far < 5e-8, "sin error on [-1000, 1000]: {far}");
        let mut arcsin: f64 = 0.0;
        for x in grid(-1.0, 1.0, 20_000) {
            let xq = quantize_to_fp(x, prec);
            arcsin = arcsin.max(error(fp_arcsin(xq, prec), dequantize_from_fp(xq, prec).asin()));
        }
        assert!(arcsin < 1e-8, "arcsin error {arcsin}");
        assert_eq!(fp_arcsin(quantize_to_fp(1.5, prec), prec), Fp::zero());
    }

    #[test]
    fn polynomial_tables_are_leading_first() {
        let atan = atan_odd_coefs_q(32);
        assert_eq!(atan.len(), ATAN_ODD_U_COEFS.len());
        assert_eq!(atan[atan.len() - 1], quantize_to_fp(ATAN_ODD_U_COEFS[0], 32));
        let sin = sin_odd_coefs_q(32);
        assert_eq!(sin.len(), SIN_ODD_U_COEFS.len());
        assert_eq!(sin[sin.len() - 1], quantize_to_fp(SIN_ODD_U_COEFS[0], 32));
    }

    #[test]
    fn fp_div_prescale_div_zero_returns_zero() {
        let precision_bits: u32 = 32;
        let aq = quantize_to_fp(5.0, precision_bits);
        let bq = Fp::zero();
        let result = fp_div_prescale(aq, bq, precision_bits);
        assert_eq!(result, Fp::zero());
    }

    fn ints(values: &[i64]) -> Vec<Fp> {
        values.iter().map(|&v| i64_to_fp(v)).collect()
    }

    #[test]
    fn signed_cmp_is_the_integer_order_on_the_window() {
        let samples = [i64::MIN, -(1 << 40), -2, -1, 0, 1, 2, 1 << 40, i64::MAX];
        for &a in &samples {
            for &b in &samples {
                assert_eq!(signed_cmp(i64_to_fp(a), i64_to_fp(b)), a.cmp(&b), "{a} vs {b}");
            }
        }
        // Total beyond the window too: p - 1 (= -1) is below 2^200.
        let big = Fp::from(2u64).pow_vartime([200u64]);
        assert_eq!(signed_cmp(-Fp::one(), big), std::cmp::Ordering::Less);
        assert_eq!(signed_cmp(-big, -Fp::one()), std::cmp::Ordering::Less);
    }

    #[test]
    fn stable_argsort_keeps_equal_values_in_original_order() {
        let values = ints(&[5, -1, 2, 0, 2, -7]);
        assert_eq!(stable_argsort(&values), vec![5, 1, 3, 2, 4, 0]);
        let values = ints(&[3, 3, 1, 3, 0, 1, 3]);
        assert_eq!(stable_argsort(&values), vec![4, 2, 5, 0, 1, 3, 6]);
    }

    #[test]
    fn hint_lanes_follow_the_argsort_and_out_of_range_ranks_give_zero() {
        use crate::ir_defs::HintKind;
        let values = ints(&[30, 10, 20, 10, -40]);
        let lane = |kind, rank: i64| fp_to_i64(hint_lane(kind, &values, i64_to_fp(rank)));
        assert_eq!((0..5).map(|r| lane(HintKind::StableArgsort, r)).collect::<Vec<_>>(), vec![4, 1, 3, 2, 0]);
        assert_eq!((0..5).map(|r| lane(HintKind::SortedValues, r)).collect::<Vec<_>>(), vec![-40, 10, 10, 20, 30]);
        assert_eq!(lane(HintKind::SortedValues, 5), 0);
        assert_eq!(lane(HintKind::StableArgsort, -1), 0);
        // Active elements only: 30, 20, -40 (flags 1, 0, 1, 0, 1).
        let mut masked = values.clone();
        masked.extend(ints(&[1, 0, 1, 0, 1]));
        let masked_lane = |rank: i64| fp_to_i64(hint_lane(HintKind::MaskedSortedValues, &masked, i64_to_fp(rank)));
        assert_eq!((0..5).map(masked_lane).collect::<Vec<_>>(), vec![-40, 20, 30, 0, 0]);
    }
}
