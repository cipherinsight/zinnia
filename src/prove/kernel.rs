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

/// Evaluate a polynomial via Horner's method: p(x) = coefs[0] + x*(coefs[1] + x*(...))
/// All values are Fp field elements in fixed-point representation (scaled by 2^precision_bits).
pub fn horner_eval(x: Fp, coefs: &[Fp], precision_bits: u32) -> Fp {
    assert!(!coefs.is_empty());
    let n = coefs.len();
    let mut acc = coefs[n - 1];
    for i in (0..n - 1).rev() {
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
/// raw = a * b; result = raw / scale; remainder = raw - result * scale.
pub fn fp_mul_rescale(a: Fp, b: Fp, precision_bits: u32) -> (Fp, Fp) {
    let raw = a * b;
    let scale = scale_fp(precision_bits);
    let scale_inv = scale.invert().unwrap();
    let result = raw * scale_inv;
    let remainder = raw - result * scale;
    (result, remainder)
}

/// Fixed-point division with pre-scaling.
/// result = (a * scale) / b
pub fn fp_div_prescale(a: Fp, b: Fp, precision_bits: u32) -> Fp {
    let scale = scale_fp(precision_bits);
    let a_scaled = a * scale;
    let b_inv = b.invert().unwrap_or(Fp::zero());
    a_scaled * b_inv
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
        // p(x) = 3 + 2*x at x=5 → 13 — precision_bits=0 for raw arithmetic
        let result = horner_eval(Fp::from(5), &[Fp::from(3), Fp::from(2)], 0);
        assert_eq!(result, Fp::from(13));
    }
}
