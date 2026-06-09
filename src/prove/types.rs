use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Proving parameters
// ---------------------------------------------------------------------------

/// Parameters that control circuit construction and proof generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvingParams {
    /// Log2 of the number of rows in the circuit (2^k rows).
    pub k: u32,
    /// Number of bits for fixed-point quantization of floats.
    pub precision_bits: u32,
    /// Number of bits for lookup-table-based range checks.
    pub lookup_bits: u32,
}

impl Default for ProvingParams {
    fn default() -> Self {
        Self {
            k: 10,
            precision_bits: 32,
            lookup_bits: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Value — the unified value type for all Python ↔ Rust exchanges
// ---------------------------------------------------------------------------

/// A dynamically-typed value that crosses the Python ↔ Rust boundary.
///
/// Used for:
/// - Circuit witness inputs (Python args → Rust prover)
/// - External function arguments (Rust preprocessor → Python callback)
/// - External function return values (Python callback → Rust preprocessor)
///
/// These are "human-readable" values — NOT field elements. The kernel
/// converts them to/from Fp as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    List(Vec<Value>),
    None,
}

// ---------------------------------------------------------------------------
// Proof artifact
// ---------------------------------------------------------------------------

/// A self-contained proof artifact produced by the prover.
///
/// # Fields
///
/// - `backend` — backend name string (e.g. `"halo2-ipa"`, `"mock"`). Selects
///   the verify-side `ProverBackend` implementation.
/// - `format` — backend-specific format discriminator (e.g.
///   `"halo2-ipa-rekeygen-v1"`, `"mock-v1"`). Lets the same backend evolve
///   its artifact layout without changing the `backend` string. Defaults to
///   the empty string for legacy artifacts produced before the discriminator
///   was added (current backends treat empty as their own v0 format).
/// - `vk_bytes` — **backend-defined VK-recovery payload**, NOT literally the
///   verifying key as bytes. The name is preserved for schema continuity, but
///   what it carries depends on `format`:
///     * `"halo2-ipa-rekeygen-v1"`: hex-encoded canonical JSON of
///       `{ "ir_stmts": [...], "precision_bits": u32, "lookup_bits": u32 }`.
///       The verifier deserializes this and calls `keygen_vk` to reconstruct
///       the `VerifyingKey` (see `compiler.halo2-cross-process-vk-serialization`
///       for the design rationale — zcash/halo2 v0.3 has no public VK serde,
///       so re-keygen on verify is the chosen path).
///     * `"mock-v1"`: empty string. Mock verify reads `proof_bytes` only.
/// - `ir_hash` — BLAKE2b-256 (32-byte) hex digest over the canonical JSON
///   bytes of the IR statement list (the `ir_stmts` array inside `vk_bytes`
///   for halo2-ipa-rekeygen-v1; empty for backends that do not carry IR).
///   **Diagnostic-only field**, NOT load-bearing for soundness — the verifier
///   re-derives this from the IR in `vk_bytes` and rejects mismatches with
///   `ProvingError::IrHashMismatch` *before* `verify_proof` runs, so the
///   fuzzer's auto-filer can classify "wrong IR shipped" distinctly from
///   "proof did not verify". The actual soundness binding between the
///   shipped IR and the proof comes from `VerifyingKey::transcript_repr`
///   being absorbed into the Fiat-Shamir transcript at both prove and verify
///   time (a load-bearing invariant of the underlying halo2 proof system; if
///   future halo2 refactors moved that bind out of the transcript, the
///   re-keygen-on-verify approach would silently lose self-binding —
///   documented near the halo2 verify path).
/// - `proof_bytes` — backend-defined proof transcript (hex for halo2,
///   a sentinel string for mock).
/// - `public_values` — public inputs as field elements (hex of the canonical
///   32-byte LE repr for halo2; `"label=value"` strings for mock).
/// - `k` — log2 of circuit row count for halo2; the same `k` from
///   `ProvingParams`. For mock, the same `params.k` is echoed but the field
///   is unused in verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofArtifact {
    pub backend: String,
    /// Backend-specific format discriminator. `#[serde(default)]` so artifacts
    /// produced by older builds (pre-discriminator) parse cleanly as
    /// `format = ""`.
    #[serde(default)]
    pub format: String,
    pub vk_bytes: String,
    /// BLAKE2b-256 hex digest of the canonical IR JSON, for diagnostic
    /// classification. `#[serde(default)]` for the same legacy-artifact
    /// reason as `format`.
    #[serde(default)]
    pub ir_hash: String,
    pub proof_bytes: String,
    pub public_values: Vec<String>,
    pub k: u32,
    /// Wall-clock duration of the **proving algorithm itself** (the halo2
    /// `create_proof` call + transcript finalize), in seconds. Excludes
    /// circuit setup (assigning advices is timed *inside* `create_proof`,
    /// so synthesis cost is captured), artifact assembly, witness JSON
    /// parsing, and the Python ↔ Rust boundary. Matches what halo2-graph's
    /// `gen_snark_ipa` measures and prints as `"Proving time: ..."`. Default
    /// 0.0 for backends that don't fill it (e.g. mock).
    #[serde(default)]
    pub prove_seconds: f64,
}

// ---------------------------------------------------------------------------
// Verify result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    pub valid: bool,
    pub error: Option<String>,
}

impl VerifyResult {
    pub fn ok() -> Self {
        Self {
            valid: true,
            error: None,
        }
    }
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self {
            valid: false,
            error: Some(reason.into()),
        }
    }
}
