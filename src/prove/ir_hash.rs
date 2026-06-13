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

/// Serialize `ir_stmts` to a *transport-canonical* JSON byte string.
///
/// `serde_json`'s `f64` serialize→parse round-trip is not always idempotent:
/// some constants drift by 1 ULP across a single cycle (observed:
/// `0.43000000000000005` → `0.4300000000000001`, two distinct bit patterns
/// that are Q32-identical, so the circuit/witness/proof are unaffected — only
/// the digest differs). Because `ir_hash` is computed by the prover over the
/// in-memory IR but recomputed by the verifier over the IR *after* it has
/// passed through the artifact's `vk_bytes` (one serialize→parse cycle), a
/// drifting constant makes the two hashes disagree and the verifier rejects an
/// otherwise-valid proof.
///
/// Iterating the round-trip to a fixed point removes the dependence on which
/// side of the wire we are on: any starting form (pre- or post-transport)
/// converges to the same canonical bytes, so prover and verifier agree.
/// Round-trip-stable IRs (the overwhelming majority — no float drift) reach
/// the fixed point on the first comparison, leaving their bytes, and hence
/// their hash, byte-for-byte identical to the pre-canonicalization behavior.
pub fn canonical_ir_bytes(ir_stmts: &[serde_json::Value]) -> Result<Vec<u8>, String> {
    let mut bytes =
        serde_json::to_vec(ir_stmts).map_err(|e| format!("ir_stmts JSON encode failed: {}", e))?;
    // A single cycle suffices in practice; the small cap is a robustness
    // backstop against a pathological value needing a second step.
    for _ in 0..8 {
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&bytes)
            .map_err(|e| format!("ir_stmts JSON canonicalize-parse failed: {}", e))?;
        let next = serde_json::to_vec(&parsed)
            .map_err(|e| format!("ir_stmts JSON re-encode failed: {}", e))?;
        if next == bytes {
            break;
        }
        bytes = next;
    }
    Ok(bytes)
}

/// Compute the canonical `ir_hash` from a slice of IR-statement JSON values.
///
/// BLAKE2b-256 (32-byte output, personalization `b"zinnia-ir-hash-1"`) over
/// the [`canonical_ir_bytes`] of `ir_stmts`, hex-encoded. This is the hash used
/// as `ProofArtifact.ir_hash` and as the halo2-IPA VK-cache key.
pub fn compute_ir_hash(ir_stmts: &[serde_json::Value]) -> Result<String, String> {
    let bytes = canonical_ir_bytes(ir_stmts)?;
    let hash = blake2b_simd::Params::new()
        .hash_length(32)
        .personal(b"zinnia-ir-hash-1")
        .hash(&bytes);
    Ok(hex_encode(hash.as_bytes()))
}
