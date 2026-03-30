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
        Self { k: 10, precision_bits: 32, lookup_bits: 8 }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofArtifact {
    pub backend: String,
    pub vk_bytes: String,
    pub proof_bytes: String,
    pub public_values: Vec<String>,
    pub k: u32,
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
    pub fn ok() -> Self { Self { valid: true, error: None } }
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self { valid: false, error: Some(reason.into()) }
    }
}
