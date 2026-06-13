//! Canonical IR-statement hashing — backend-agnostic core helper.
//!
//! The `ir_hash` is a BLAKE2b-256 digest over `serde_json::to_vec(&ir_stmts)`,
//! hex-encoded for transport. It is computed eagerly on the always-compiled
//! prove path (cached on `CompiledIR`) because the halo2-IPA backend carries
//! it in `ProofArtifact.ir_hash` and uses it as the VK-cache key, and because
//! re-serializing + re-hashing a 40k-statement IR on every prove call is
//! ~200 ms of overhead for output-heavy kernels.
//!
//! It lives here, in the always-compiled core, rather than under
//! `src/prove/halo2/` so that the `--features noir-backend` (and mock-only)
//! builds — which never link `halo2_proofs` — can still compute it. The
//! halo2-IPA `artifact_format::compute_ir_hash` delegates here, so the
//! on-the-wire hash is byte-for-byte identical to before this seam existed.
//!
//! Only `blake2b_simd` is needed (a light, fast crate already present in the
//! tree via `halo2curves`); no proving-system dependency.

/// Hex-encode a byte slice (lowercase, two chars per byte).
pub fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Compute the canonical `ir_hash` from a slice of IR-statement JSON values.
///
/// BLAKE2b-256 (32-byte output, personalization `b"zinnia-ir-hash-1"`) over
/// `serde_json::to_vec(&ir_stmts)`, hex-encoded. This is the hash used as
/// `ProofArtifact.ir_hash` and as the halo2-IPA VK-cache key.
pub fn compute_ir_hash(ir_stmts: &[serde_json::Value]) -> Result<String, String> {
    let bytes =
        serde_json::to_vec(ir_stmts).map_err(|e| format!("ir_stmts JSON encode failed: {}", e))?;
    let hash = blake2b_simd::Params::new()
        .hash_length(32)
        .personal(b"zinnia-ir-hash-1")
        .hash(&bytes);
    Ok(hex_encode(hash.as_bytes()))
}
