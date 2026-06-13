//! Fixed-point field arithmetic helpers and the core field-type seam.
//!
//! Floats are represented as fixed-point integers in the prime field:
//!
//!   x_quantized = round(x * 2^PRECISION_BITS)
//!
//! This preserves ordering and allows field-native add/sub while requiring
//! rescaling after multiplication.
//!
//! # The `F` seam
//!
//! [`F`] is the single field type used by the always-compiled core
//! (`ResolvedWitness`, the shared `kernel`, the mock synthesizer, and the
//! preprocessor). It aliases `halo2curves::bn256::Fr` — the BN254 scalar
//! field — pulled from the *light* `halo2curves` crate rather than the full
//! `halo2_proofs` proving system. Because `halo2_proofs` v0.3.0 depends on
//! this same `halo2curves` version, `F` is ABI-identical to the `Fr` the
//! halo2 backends constrain, so no conversion happens at the backend
//! boundary. Routing the core through this alias is what lets
//! `halo2_proofs` be a genuinely optional dependency: a `--features
//! noir-backend` build never links it.

/// The proving field: BN254's scalar field, `halo2curves::bn256::Fr`.
///
/// This is the one place the concrete curve is named for the core. The two
/// halo2 backends use `halo2_proofs::halo2curves::bn256::Fr`, which resolves
/// to this identical type (same `halo2curves` crate version).
pub type F = halo2curves::bn256::Fr;

/// Re-export of the `ff::Field` trait so core modules can call `.invert()`,
/// `Field::zero()`, etc. on [`F`] without naming `halo2curves` directly.
/// Equivalent to `halo2_proofs::arithmetic::Field` (that is just `ff::Field`).
pub use halo2curves::ff::Field;

/// Re-export of `ff::PrimeField` for `.to_repr()` / `from_repr` byte access.
pub use halo2curves::ff::PrimeField;

/// Quantize a float to a fixed-point integer representation.
///
/// Returns the quantized value as an `i128` (signed, before reduction to field).
pub fn quantize(value: f64, precision_bits: u32) -> i128 {
    let scale = (1u128 << precision_bits) as f64;
    (value * scale).round() as i128
}

/// Dequantize a fixed-point integer back to a float.
pub fn dequantize(quantized: i128, precision_bits: u32) -> f64 {
    let scale = (1u128 << precision_bits) as f64;
    quantized as f64 / scale
}

/// The quantization scale factor as a u128.
pub fn quantization_scale(precision_bits: u32) -> u128 {
    1u128 << precision_bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let precision = 32;
        let values = [0.0, 1.0, -1.0, 3.14159, -2.71828, 0.001, 1000.5];
        for &v in &values {
            let q = quantize(v, precision);
            let d = dequantize(q, precision);
            assert!(
                (d - v).abs() < 1e-6,
                "Round-trip failed for {}: got {}",
                v,
                d
            );
        }
    }

    #[test]
    fn test_quantize_preserves_ordering() {
        let precision = 32;
        let a = quantize(1.5, precision);
        let b = quantize(2.5, precision);
        assert!(a < b);
    }

    #[test]
    fn test_quantization_scale() {
        assert_eq!(quantization_scale(0), 1);
        assert_eq!(quantization_scale(1), 2);
        assert_eq!(quantization_scale(32), 1 << 32);
    }
}
