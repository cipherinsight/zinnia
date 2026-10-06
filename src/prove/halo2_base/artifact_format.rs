//! Canonical wire format for the halo2-base artifact payload.

use serde::{Deserialize, Serialize};

use halo2_base::gates::{
    circuit::BaseCircuitParams, flex_gate::MultiPhaseThreadBreakPoints,
};

/// halo2-base cannot reconstruct a witness-free circuit via
/// `without_witnesses`; its builder intentionally leaves that method
/// unimplemented. The artifact therefore carries the auto-computed layout
/// pinning so verify can rebuild the exact VK shape from the IR.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Halo2BasePinnedRekeygenV1Payload {
    pub ir_stmts: Vec<serde_json::Value>,
    pub precision_bits: u32,
    pub lookup_bits: u32,
    pub base_params: BaseCircuitParams,
    pub break_points: MultiPhaseThreadBreakPoints,
    /// The circuit's verifier challenges (`circuit::CircuitParams`); 0 for
    /// a one-phase circuit (and in artifacts written before there were any).
    #[serde(default)]
    pub challenges: usize,
    /// Key column sets per memory-table group (`circuit::CircuitParams`).
    #[serde(default)]
    pub table_layout: Vec<usize>,
    /// RAM traces (`circuit::CircuitParams`).
    #[serde(default)]
    pub ram_traces: usize,
    /// Graph-phase provenance records (Step 2: synthesis consumes the
    /// table — interval/affine narrowing changes the emitted gate
    /// count). Verify re-synthesizes from `ir_stmts`; without the
    /// table it builds a DIFFERENT circuit and the pinned break points
    /// mismatch. `default` keeps old artifacts decodable (they fail
    /// verify on provenance-dependent circuits either way).
    #[serde(default)]
    pub provenance: crate::provenance::ProvenanceWire,
}

pub const HALO2_BASE_PINNED_REKEYGEN_V1: &str = "halo2-base-pinned-rekeygen-v1";

pub fn compute_ir_hash(ir_stmts: &[serde_json::Value]) -> Result<String, String> {
    // Transport-canonical bytes so the prover's pre-`vk_bytes` hash matches the
    // verifier's post-`vk_bytes` recompute even when an `f64` constant drifts
    // 1 ULP across serde_json's serialize→parse cycle. See
    // `crate::prove::ir_hash::canonical_ir_bytes`.
    let canonical = crate::prove::ir_hash::canonical_ir_bytes(ir_stmts)?;
    let hash = blake2b_simd::Params::new()
        .hash_length(32)
        .hash(&canonical);
    Ok(hash.as_bytes().iter().map(|b| format!("{:02x}", b)).collect())
}

pub fn encode(payload: &Halo2BasePinnedRekeygenV1Payload) -> Result<String, String> {
    let bytes = serde_json::to_vec(payload).map_err(|e| e.to_string())?;
    Ok(hex_encode(&bytes))
}

pub fn decode(s: &str) -> Result<Halo2BasePinnedRekeygenV1Payload, String> {
    let bytes = hex_decode(s)?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

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
