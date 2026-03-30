use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};
use crate::circuit_input::{ResolvedWitness, InputPath};

// ---------------------------------------------------------------------------
// ProverBackend — top-level backend trait
// ---------------------------------------------------------------------------

/// A pluggable ZK proving backend.
///
/// Each backend (mock, halo2-ipa, future circom, etc.) implements this trait
/// to provide the full prove/verify lifecycle. The trait is object-safe so
/// backends can be dispatched dynamically via `Box<dyn ProverBackend>`.
pub trait ProverBackend: Send + Sync {
    /// A human-readable identifier for this backend (e.g., "halo2-ipa").
    fn name(&self) -> &'static str;

    /// Analyze the IR to recommend circuit-size parameters.
    fn estimate_params(&self, ir: &IRGraph) -> Result<ProvingParams, ProvingError>;

    /// Generate a proof from an IR graph and resolved witness.
    fn prove(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> Result<ProofArtifact, ProvingError>;

    /// Verify a self-contained proof artifact.
    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError>;
}

// ---------------------------------------------------------------------------
// Synthesizer — chip-level dispatch trait
// ---------------------------------------------------------------------------

/// Abstracts the constraint-generation backend from the IR interpreter.
///
/// The IR interpreter walks the `IRGraph` and calls these methods for each
/// IR instruction. Implementations include:
///
/// - `MockSynthesizer`: evaluates operations with plain arithmetic (testing)
/// - `Halo2Synthesizer`: assigns witnesses and creates halo2 constraints
///
/// The `CellRef` associated type is opaque to the interpreter:
/// - For mock: a concrete field value
/// - For halo2: an `AssignedCell` handle
pub trait Synthesizer {
    /// An opaque reference to a cell/value in the circuit.
    type CellRef: Clone;

    // ── Constants ─────────────────────────────────────────────────────

    fn constant_int(&mut self, value: i64) -> Result<Self::CellRef, ProvingError>;
    fn constant_float(&mut self, value: f64) -> Result<Self::CellRef, ProvingError>;
    fn constant_bool(&mut self, value: bool) -> Result<Self::CellRef, ProvingError>;

    // ── Integer arithmetic ────────────────────────────────────────────

    fn add_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sub_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn mul_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn div_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn floor_div_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn mod_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn pow_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn abs_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sign_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn inv_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Float arithmetic (fixed-point) ────────────────────────────────

    fn add_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sub_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn mul_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn floor_div_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn mod_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn pow_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn abs_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sign_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Integer comparisons ───────────────────────────────────────────

    fn eq_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn ne_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn lt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn lte_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn gt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn gte_i(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Float comparisons ─────────────────────────────────────────────

    fn eq_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn ne_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn lt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn lte_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn gt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn gte_f(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Transcendentals ───────────────────────────────────────────────

    fn sin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sinh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn cos_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn cosh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn tan_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn tanh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sqrt_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn exp_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn log_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Boolean logic ─────────────────────────────────────────────────

    fn logical_and(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn logical_or(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn logical_not(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Selection (mux) ───────────────────────────────────────────────

    fn select(
        &mut self,
        cond: &Self::CellRef,
        if_true: &Self::CellRef,
        if_false: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;

    // ── Casting ───────────────────────────────────────────────────────

    fn int_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn float_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn bool_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── I/O ───────────────────────────────────────────────────────────

    fn read_input(&mut self, path: &InputPath, is_public: bool) -> Result<Self::CellRef, ProvingError>;
    fn read_external_result(&mut self, store_idx: u32, output_idx: u32) -> Result<Self::CellRef, ProvingError>;
    fn expose_public(&mut self, a: &Self::CellRef, label: &str) -> Result<(), ProvingError>;
    fn assert_true(&mut self, a: &Self::CellRef) -> Result<(), ProvingError>;

    // ── Memory ────────────────────────────────────────────────────────

    fn allocate_memory(&mut self, segment_id: u32, size: u32, init: i64) -> Result<(), ProvingError>;
    fn write_memory(
        &mut self,
        segment_id: u32,
        addr: &Self::CellRef,
        value: &Self::CellRef,
    ) -> Result<(), ProvingError>;
    fn read_memory(&mut self, segment_id: u32, addr: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn memory_trace_emit(
        &mut self,
        segment_id: u32,
        is_write: bool,
        args: &[Self::CellRef],
    ) -> Result<(), ProvingError>;
    fn memory_trace_seal(&mut self) -> Result<(), ProvingError>;

    // ── Dynamic NDArray ───────────────────────────────────────────────

    fn allocate_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        dtype_name: &str,
        max_length: u32,
        max_rank: u32,
    ) -> Result<(), ProvingError>;
    fn witness_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        args: &[Self::CellRef],
    ) -> Result<(), ProvingError>;
    fn assert_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        max_length: u32,
        args: &[Self::CellRef],
    ) -> Result<(), ProvingError>;
    fn dynamic_ndarray_get_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        index: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn dynamic_ndarray_set_item(
        &mut self,
        array_id: u32,
        segment_id: u32,
        index: &Self::CellRef,
        value: &Self::CellRef,
    ) -> Result<(), ProvingError>;

    // ── Hash ──────────────────────────────────────────────────────────

    fn poseidon_hash(&mut self, inputs: &[Self::CellRef]) -> Result<Self::CellRef, ProvingError>;
    fn eq_hash(&mut self, a: &Self::CellRef, b: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
}
