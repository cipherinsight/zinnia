//! Shared computation kernel for all proving backends.
//!
//! This module provides pure functions for field arithmetic, fixed-point
//! quantization, and transcendental function evaluation that are used
//! identically by both the mock and halo2 synthesizers. This ensures
//! semantic consistency: the mock backend produces the exact same values
//! that the halo2 backend constrains.
//!
//! All functions operate on `pasta_curves::Fp` — the Pasta base field
//! used by zcash/halo2 with IPA commitment.

use pasta_curves::Fp;
use pasta_curves::group::ff::PrimeField;

/// Re-export the Field trait so backends can call `.invert()` etc.
pub use halo2_proofs::arithmetic::Field;

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
/// Even-indexed coefficients are near-zero (atan is odd).
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

/// sin(x) on [0, pi), degree 14.
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
        -Fp::from((-v) as u64)
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
        -(low as i64)
    } else {
        let bytes = v.to_repr();
        let low = u64::from_le_bytes(bytes.as_ref()[0..8].try_into().unwrap());
        low as i64
    }
}

/// Check if a field element represents a negative value (> (p-1)/2).
///
/// The Pasta Fp field has p ≈ 2^254. The "negative" half is [(p+1)/2, p-1].
/// We compute (p-1)/2 and compare using big-endian byte ordering.
pub fn fp_is_negative(v: Fp) -> bool {
    if v == Fp::zero() {
        return false;
    }
    // half = (p-1)/2. In the Pasta field, -1 = p-1, so (p-1)/2 = (-1) * 2^{-1}.
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

use crate::prove::types::Value;
use crate::prove::error::ProvingError;

/// Parse a 64-char hex string into an Fp field element.
pub fn hex_str_to_fp(s: &str) -> Result<Fp, ProvingError> {
    if s.len() != 64 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ProvingError::other(format!("Invalid hex field element: '{}'", s)));
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
        Value::List(_) => Err(ProvingError::other("Cannot convert list to a single field element")),
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
/// NOTE: This matches the layout of `LOG2_COEFS`, `EXP2_COEFS`, `SIN_COEFS`, and
/// `ATAN_COEFS` in this module — they are all stored leading-first. A previous
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
    let scale = 1i128 << precision_bits;
    let quotient = raw.div_euclid(scale);
    let remainder = raw.rem_euclid(scale); // 0 <= remainder < scale
    let result_fp = i128_to_fp(quotient);
    let remainder_fp = Fp::from(remainder as u64); // remainder is non-negative and < 2^precision_bits
    (result_fp, remainder_fp)
}

/// Fixed-point division with pre-scaling.
///
/// Computes `result = floor((a * scale) / b)` where `scale = 2^precision_bits`.
/// Uses i128 integer division so the result is value-correct (the previous
/// implementation used field modular inverse and produced wrap-around garbage
/// for non-trivial inputs). For divide-by-zero, returns zero (matching the prior
/// `b.invert().unwrap_or(Fp::zero())` behavior).
pub fn fp_div_prescale(a: Fp, b: Fp, precision_bits: u32) -> Fp {
    let b_signed = fp_to_i64(b) as i128;
    if b_signed == 0 {
        return Fp::zero();
    }
    let a_signed = fp_to_i64(a) as i128;
    let scale = 1i128 << precision_bits;
    let numerator = a_signed * scale;
    let quotient = numerator.div_euclid(b_signed);
    i128_to_fp(quotient)
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

/// Floor division: returns (quotient, remainder) such that a = b*q + r.
pub fn fp_floor_div(a: Fp, b: Fp) -> (Fp, Fp) {
    let sa = fp_to_i64(a);
    let sb = fp_to_i64(b);
    if sb == 0 {
        return (Fp::zero(), Fp::zero());
    }
    let q = sa.div_euclid(sb);
    let r = sa.rem_euclid(sb);
    (i64_to_fp(q), Fp::from(r as u64))
}

/// Compute sin(x) using native f64 math on the dequantized value.
pub fn fp_sin(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.sin(), precision_bits)
}

/// Compute exp(x) using native f64 math on the dequantized value.
pub fn fp_exp(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.exp(), precision_bits)
}

/// Compute ln(x) using native f64 math on the dequantized value.
pub fn fp_log(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.ln(), precision_bits)
}

/// Compute cos(x) using native f64 math on the dequantized value.
pub fn fp_cos(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.cos(), precision_bits)
}

/// Compute sqrt(x) using native f64 math on the dequantized value.
pub fn fp_sqrt(x: Fp, precision_bits: u32) -> Fp {
    let v = dequantize_from_fp(x, precision_bits);
    quantize_to_fp(v.sqrt(), precision_bits)
}

/// Compute sinh: (exp(x) - exp(-x)) / 2.
pub fn fp_sinh(x: Fp, precision_bits: u32) -> Fp {
    let ex = fp_exp(x, precision_bits);
    let enx = fp_exp(-x, precision_bits);
    let diff = ex - enx;
    let two = quantize_to_fp(2.0, precision_bits);
    fp_div_prescale(diff, two, precision_bits)
}

/// Compute cosh: (exp(x) + exp(-x)) / 2.
pub fn fp_cosh(x: Fp, precision_bits: u32) -> Fp {
    let ex = fp_exp(x, precision_bits);
    let enx = fp_exp(-x, precision_bits);
    let sum = ex + enx;
    let two = quantize_to_fp(2.0, precision_bits);
    fp_div_prescale(sum, two, precision_bits)
}

/// Compute tan: sin(x) / cos(x).
pub fn fp_tan(x: Fp, precision_bits: u32) -> Fp {
    let s = fp_sin(x, precision_bits);
    let c = fp_cos(x, precision_bits);
    fp_div_prescale(s, c, precision_bits)
}

/// Compute tanh: sinh(x) / cosh(x).
pub fn fp_tanh(x: Fp, precision_bits: u32) -> Fp {
    let s = fp_sinh(x, precision_bits);
    let c = fp_cosh(x, precision_bits);
    fp_div_prescale(s, c, precision_bits)
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
            assert!((d - v).abs() < 1e-6, "Round-trip failed for {}: got {}", v, d);
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
            0, 1, -1, 42, -42,
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
                assert_eq!(fp_to_i64(fp), v as i64, "i64-range round-trip failed for {}", v);
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
                rem_i128, scale,
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
            assert_eq!(
                recomputed, raw,
                "div_mod identity failed for ({a}, {b})",
            );
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
    fn fp_div_prescale_div_zero_returns_zero() {
        let precision_bits: u32 = 32;
        let aq = quantize_to_fp(5.0, precision_bits);
        let bq = Fp::zero();
        let result = fp_div_prescale(aq, bq, precision_bits);
        assert_eq!(result, Fp::zero());
    }
}
