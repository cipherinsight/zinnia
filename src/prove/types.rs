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
/// - `backend` — backend name string (e.g. `"halo2-base"`, `"noir"`,
///   `"mock"`). Selects the verify-side `ProverBackend` implementation.
/// - `format` — backend-specific format discriminator (e.g.
///   `"halo2-base-pinned-rekeygen-v1"`, `"mock-v1"`). Lets the same backend
///   evolve its artifact layout without changing the `backend` string.
///   Defaults to the empty string for artifacts written before the
///   discriminator was added.
/// - `vk_bytes` — **backend-defined VK-recovery payload**, not the verifying
///   key itself. For `"halo2-base-pinned-rekeygen-v1"` it is the hex-encoded
///   JSON of `halo2_base::artifact_format::Halo2BasePinnedRekeygenV1Payload`
///   (the IR statements, precision and lookup bits and the pinned circuit
///   layout), from which the verifier rebuilds the verifying key by keygen.
///   For `"mock-v1"` it is empty: mock verify reads `proof_bytes` only.
/// - `ir_hash` — BLAKE2b-256 hex digest over the canonical JSON bytes of the
///   IR statement list. Diagnostic only: the verifier re-derives it from the
///   IR in `vk_bytes` and rejects a mismatch with
///   `ProvingError::IrHashMismatch` before verifying, so "wrong IR shipped"
///   is reported apart from "proof did not verify". The binding between the
///   shipped IR and the proof comes from the verifying key being absorbed
///   into the Fiat-Shamir transcript at prove and verify time.
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
