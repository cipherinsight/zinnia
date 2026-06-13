//! Canonical wire format for the halo2-ipa-rekeygen-v1 `vk_bytes` payload.
//!
//! The chosen direction for `compiler.halo2-cross-process-vk-serialization`
//! (option D' — re-keygen on verify with `(ir_hash, params) → VK` cache)
//! ships the IR through `ProofArtifact.vk_bytes` and re-runs `keygen_vk`
//! on the verify side. zcash/halo2 v0.3 has no public `VerifyingKey`
//! serde — see the card README and the tradeoff writeup for the full
//! landscape and rationale.
//!
//! # Wire format
//!
//! ```text
//! vk_bytes = hex( serde_json::to_vec(&Halo2IpaRekeygenV1Payload {
//!     ir_stmts: [IRStatement::export(), ...],
//!     precision_bits: u32,
//!     lookup_bits: u32,
//! }) )
//! ```
//!
//! - `ir_stmts` is the IRGraph's full statement list in the same canonical
//!   JSON format already used by `IRGraph::export_stmts` / `import_stmts`
//!   (the same path Python uses for `export_ir_json` / `import_ir_json`,
//!   so the encoding is shared with a tested codepath).
//! - `precision_bits` and `lookup_bits` are the `ProvingParams` components
//!   that affect the circuit's constraint-system shape beyond `k` (`k`
//!   lives at the top of `ProofArtifact` already, so we don't duplicate
//!   it here — it's read from `artifact.k` on verify).
//! - Outer encoding is `serde_json::to_vec` (no whitespace, default
//!   `BTreeMap`-backed `serde_json::Map` → lexicographically sorted keys),
//!   then hex. JSON over bincode for two reasons: human-debuggable
//!   artifacts on disk, and reuse of the existing IR JSON round-trip.
//!   Swappable to bincode later if @Vera's sweep telemetry surfaces an
//!   artifact-size or parse-cost regression.
//!
//! # ir_hash and the Fiat-Shamir self-binding invariant
//!
//! `ProofArtifact.ir_hash` is BLAKE2b-256 over the canonical JSON bytes
//! of the `ir_stmts` array only (NOT over `precision_bits` / `lookup_bits`;
//! @Sage's framing of "the IR hash" stays clean as "did the verifier
//! receive the same IR the prover proved"). The verifier recomputes
//! `ir_hash` from the IR in `vk_bytes` and rejects mismatches with
//! `ProvingError::IrHashMismatch` *before* `verify_proof` runs.
//!
//! **Soundness of D' depends on `VerifyingKey::transcript_repr` being
//! absorbed into the Fiat-Shamir transcript at both prove and verify
//! time** (this is the standard halo2 pattern in both zcash v0.3 and PSE
//! halo2; verified by reading `VerifyingKey::hash_into` in
//! `halo2_proofs-0.3.2/src/plonk.rs`). Under that invariant, a malicious
//! prover shipping mismatched IR causes the verifier's re-keygen to
//! produce a different VK, the F-S transcript diverges at the VK-bind
//! step inside `verify_proof`, and the proof rejects. The `ir_hash`
//! field is therefore a **diagnostic affordance** — it gives the
//! differential fuzzer's auto-filer a clean "wrong IR shipped"
//! classification rather than a generic "transcript mismatch at challenge
//! N" — not a soundness gate. (Equivalent risk to option A: any future
//! halo2 refactor that moves the VK-bind out of the transcript would
//! silently lose self-binding regardless of which direction we picked.)
//!
//! # IR-export determinism precondition
//!
//! The `ir_hash` field is byte-stable across processes iff
//! `IRStatement::export` produces deterministic JSON bytes for a fixed
//! IR. Audit (see thread `#zinnia:7690a720`):
//!
//! - All `IR::export_data` variants use `serde_json::json!` with
//!   fixed-source-order keys; the resulting `serde_json::Map` is
//!   `BTreeMap`-backed (no `preserve_order` feature on `serde_json` in
//!   this project — confirmed via Cargo.lock, no `indexmap`
//!   transitive dep), so keys serialize lexicographically.
//! - The one variant with a `HashMap<String, serde_json::Value>` field
//!   (`IR::InvokeExternal::kwargs`) is converted to `serde_json::Value`
//!   via `From<HashMap<K, V>> for Value`, which collects into the
//!   `BTreeMap`-backed `Map`, so the eventual JSON byte order is
//!   alphabetical — deterministic across processes.
//! - `IRStatement::export` (`stmt_id`, `ir_instance`, `arguments`) and
//!   `IRGraph::export_stmts` (Vec iteration in stable order) preserve
//!   that determinism.
//!
//! This is a load-bearing invariant for D' (a regression would silently
//! make `ir_hash` mismatch across processes for the same IR). Pinned by:
//! 1. The `serde_json::json!` source-order convention in
//!    `IRStatement::export` / `IR::export_data`.
//! 2. A determinism regression test in `prove/halo2/tests.rs` that
//!    builds an `InvokeExternal` with multiple `kwargs` keys, exports
//!    twice, and asserts byte-equality.
//! 3. @Sage's `keygen_vk` byte-equality test one layer up — even if the
//!    determinism precondition silently breaks, the byte-equality guard
//!    on `pinned()` of two keygen runs catches it.

use serde::{Deserialize, Serialize};

/// Canonical struct for the `halo2-ipa-rekeygen-v1` payload carried inside
/// `ProofArtifact.vk_bytes`. Field names are stable: changing them breaks
/// cross-version artifact compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halo2IpaRekeygenV1Payload {
    /// Full IR statement list, in the same canonical JSON shape produced
    /// by `IRGraph::export_stmts`. Round-trips through
    /// `IRGraph::import_stmts` on the verify side.
    pub ir_stmts: Vec<serde_json::Value>,
    /// Fixed-point precision for float operations. Read by the
    /// Halo2Synthesizer during keygen-time `synthesize` invocation.
    pub precision_bits: u32,
    /// Range-check lookup-table bit width. Read by the Halo2Synthesizer
    /// during keygen-time `synthesize` invocation.
    pub lookup_bits: u32,
}

/// Discriminator string for the `format` field on `ProofArtifact` when the
/// halo2-ipa backend uses the D' re-keygen path.
pub const HALO2_IPA_REKEYGEN_V1: &str = "halo2-ipa-rekeygen-v1";

/// Encode the payload to the hex-of-canonical-JSON wire form.
pub fn encode(payload: &Halo2IpaRekeygenV1Payload) -> Result<String, String> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|e| format!("Halo2IpaRekeygenV1Payload JSON encode failed: {}", e))?;
    Ok(hex_encode(&bytes))
}

/// Decode a hex-of-canonical-JSON payload back to the typed struct.
pub fn decode(s: &str) -> Result<Halo2IpaRekeygenV1Payload, String> {
    let bytes = hex_decode(s)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| format!("Halo2IpaRekeygenV1Payload JSON decode failed: {}", e))
}

/// Compute the `ir_hash` field from a slice of IR-statement JSON values.
/// BLAKE2b-256 (32-byte output) over `serde_json::to_vec(&ir_stmts)`,
/// hex-encoded for transport. This is the canonical hash used both as the
/// `ProofArtifact.ir_hash` field and as the cache key for the VK cache.
pub fn compute_ir_hash(ir_stmts: &[serde_json::Value]) -> Result<String, String> {
    // Delegate to the always-compiled core helper so the wire hash is shared
    // with the eager `CompiledIR::cached_artifact_inputs` path and cannot
    // drift. Output is byte-for-byte identical (same BLAKE2b-256 +
    // `personal(b"zinnia-ir-hash-1")`).
    crate::prove::ir_hash::compute_ir_hash(ir_stmts)
}

// ---- Local hex helpers (matching the rest of the halo2 backend) ----------

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("Odd-length hex string".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("Invalid hex: {}", e)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_round_trips_byte_for_byte() {
        let payload = Halo2IpaRekeygenV1Payload {
            ir_stmts: vec![serde_json::json!({"stmt_id": 0, "kind": "ConstantInt", "value": 10})],
            precision_bits: 32,
            lookup_bits: 8,
        };
        let encoded = encode(&payload).unwrap();
        let decoded = decode(&encoded).unwrap();
        let re_encoded = encode(&decoded).unwrap();
        assert_eq!(
            encoded, re_encoded,
            "Round-trip encode must be byte-stable for ir_hash determinism"
        );
    }

    #[test]
    fn ir_hash_is_stable_across_invocations() {
        let stmts = vec![
            serde_json::json!({"stmt_id": 0, "kind": "ConstantInt", "value": 10}),
            serde_json::json!({"stmt_id": 1, "kind": "ConstantInt", "value": 20}),
        ];
        let h1 = compute_ir_hash(&stmts).unwrap();
        let h2 = compute_ir_hash(&stmts).unwrap();
        assert_eq!(h1, h2);
        // BLAKE2b-256 → 32 bytes → 64 hex chars.
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn ir_hash_differs_on_different_ir() {
        let a = compute_ir_hash(&[serde_json::json!({"stmt_id": 0, "value": 1})]).unwrap();
        let b = compute_ir_hash(&[serde_json::json!({"stmt_id": 0, "value": 2})]).unwrap();
        assert_ne!(a, b);
    }
}
