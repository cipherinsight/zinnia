use crate::circuit_input::{InputPath, ResolvedWitness};
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

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

    /// Generate a proof using caller-provided pre-serialized IR statements
    /// and ir_hash. For backends that need them on every prove call (the
    /// halo2-IPA path uses them to populate the artifact for cross-process
    /// verify), this skips the per-call re-serialize / re-hash and lets the
    /// caller cache them once at compile time.
    ///
    /// The default implementation re-derives them on every call — backends
    /// that benefit from the cache (halo2) override this.
    fn prove_with_cached_artifact(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
        _ir_stmts: &[serde_json::Value],
        _ir_hash: &str,
    ) -> Result<ProofArtifact, ProvingError> {
        self.prove(ir, witness, params)
    }

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

    /// Optional hook called by `interpret_ir` once per IR statement, before
    /// dispatching the matching trait method. Backends may override to
    /// attach per-statement metadata (e.g. the Noir backend looks up a
    /// pre-computed fits-u32 bound for the result of this statement).
    /// Default is a no-op.
    fn on_stmt_begin(&mut self, _stmt_id: u32) {}

    /// Optional hook called by `interpret_ir` after a statement produced a
    /// cell, with the statement's stable `ValueId`. Backends use it to
    /// maintain a ValueId ↔ cell correspondence for consuming the
    /// structural-provenance table (`src/provenance.rs`). Default no-op.
    fn on_stmt_result(
        &mut self,
        _stmt_id: u32,
        _value_id: crate::types::ValueId,
        _cell: &Self::CellRef,
    ) {
    }

    // ── Constants ─────────────────────────────────────────────────────

    fn constant_int(&mut self, value: i64) -> Result<Self::CellRef, ProvingError>;
    fn constant_float(&mut self, value: f64) -> Result<Self::CellRef, ProvingError>;
    fn constant_bool(&mut self, value: bool) -> Result<Self::CellRef, ProvingError>;

    // ── Integer arithmetic ────────────────────────────────────────────

    fn add_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn sub_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn mul_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn div_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn floor_div_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    /// Euclidean floor division with a producer-certified bound: `a`, `b`, the
    /// quotient and the remainder all fit in `divisor_bits` bits (see
    /// [`crate::ir::IR::FloorDivIPinned`]). The default ignores the bound and
    /// delegates to [`Self::floor_div_i`] — correct for every backend whose
    /// `floor_div_i` is already sound. Backends with an expensive general
    /// division (the Noir Field path) override this to emit a cheaper
    /// bit-decomposition-range-checked form.
    fn floor_div_i_pinned(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
        _divisor_bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        self.floor_div_i(a, b)
    }
    fn mod_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn pow_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn abs_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sign_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn inv_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Integer bitwise ───────────────────────────────────────────────

    fn bit_and_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn bit_or_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn bit_xor_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn shl_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn shr_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn bit_not_i(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Float arithmetic (fixed-point) ────────────────────────────────

    fn add_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn sum_f(&mut self, terms: &[Self::CellRef]) -> Result<Self::CellRef, ProvingError> {
        if terms.is_empty() {
            return self.constant_float(0.0);
        }
        let mut acc = terms[0].clone();
        for term in &terms[1..] {
            acc = self.add_f(&acc, term)?;
        }
        Ok(acc)
    }
    fn sub_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn mul_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    /// Static float product-sum over adjacent operand pairs.
    ///
    /// `pairs` contains `(lhs, rhs)` operands in frontend order
    /// `[lhs0, rhs0, lhs1, rhs1, ...]` after interpreter validation.
    /// Each product must use the same fixed-point Q-rescale semantics as
    /// scalar `mul_f`; accumulation uses existing float-add semantics.
    fn product_sum_f(
        &mut self,
        pairs: &[(Self::CellRef, Self::CellRef)],
    ) -> Result<Self::CellRef, ProvingError>;
    fn div_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn floor_div_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn mod_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn pow_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    /// Float base raised to an Integer exponent. Backends that don't
    /// special-case integer exponents may fall back to `pow_f` after a
    /// float-cast on the exponent, but lowering as integer-exponent
    /// square-and-multiply avoids `exp(b * log(a))` overhead.
    fn pow_f_int(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError> {
        let bf = self.float_cast(b)?;
        self.pow_f(a, &bf)
    }
    fn abs_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sign_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Integer comparisons ───────────────────────────────────────────

    fn eq_i(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn ne_i(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn lt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    /// `lt_i` for operands that are provably non-negative and fit in u128.
    /// Backends that can take advantage of the bound (e.g. Noir's native
    /// `<` on u128) override; others fall through to `lt_i` and lose
    /// nothing.
    fn lt_i_bounded(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError> {
        self.lt_i(a, b)
    }
    /// `gt_i` companion to `lt_i_bounded` with the same operand contract.
    fn gt_i_bounded(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError> {
        self.gt_i(a, b)
    }
    /// Multi-axis bounded-range mask: returns Field 0/1 = 1 iff for every
    /// axis k: `lo_k <= idx_k < hi_k`. `triples[k] = (idx, lo, hi)`. Same
    /// operand contract as `lt_i_bounded`. Default implementation falls
    /// back to a per-axis `lt_i_bounded + ge` + `logical_and` chain;
    /// backends that can fuse the conjunction (e.g. Noir's combined `&`
    /// on u128 casts sharing a single RANGE table) override.
    fn index_in_box(
        &mut self,
        triples: &[(Self::CellRef, Self::CellRef, Self::CellRef)],
    ) -> Result<Self::CellRef, ProvingError> {
        // Default: emit per-axis `lo <= idx < hi` + a logical-and chain.
        let mut acc: Option<Self::CellRef> = None;
        for (idx, lo, hi) in triples {
            let lt = self.lt_i_bounded(idx, hi)?;
            let ge_lo = {
                let lt_lo = self.lt_i_bounded(idx, lo)?;
                self.logical_not(&lt_lo)?
            };
            let in_axis = self.logical_and(&ge_lo, &lt)?;
            acc = Some(match acc {
                None => in_axis,
                Some(prev) => self.logical_and(&prev, &in_axis)?,
            });
        }
        // No axes → vacuously true.
        match acc {
            Some(c) => Ok(c),
            None => self.constant_int(1),
        }
    }
    fn lte_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn gt_i(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn gte_i(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;

    // ── Float comparisons ─────────────────────────────────────────────

    fn eq_f(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn ne_f(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn lt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn lte_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn gt_f(&mut self, a: &Self::CellRef, b: &Self::CellRef)
        -> Result<Self::CellRef, ProvingError>;
    fn gte_f(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;

    // ── Transcendentals ───────────────────────────────────────────────

    fn sin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sinh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn cos_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn cosh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn tan_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn tanh_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn sqrt_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    /// `IR::SqrtProductIF`: `floor(sqrt(b * c) * 2^P)` of two integers, with
    /// no proof unless `0 <= b < 2^64` and `0 <= b * c < 2^sqrt_int_bits(P)`
    /// (`kernel::fp_sqrt_product_int`).
    fn sqrt_product_i_f(&mut self, _b: &Self::CellRef, _c: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Err(ProvingError::synthesis("this backend does not support the square root of an integer product"))
    }
    fn exp_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn log_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn arcsin_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn arccos_f(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn arctan2_f(
        &mut self,
        y: &Self::CellRef,
        x: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;

    // ── Boolean logic ─────────────────────────────────────────────────

    fn logical_and(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn logical_or(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn logical_not(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── Selection (mux) ───────────────────────────────────────────────

    fn select(
        &mut self,
        cond: &Self::CellRef,
        if_true: &Self::CellRef,
        if_false: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    fn select_bit(
        &mut self,
        cond: &Self::CellRef,
        if_true: &Self::CellRef,
        if_false: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError> {
        self.assert_bit(cond)?;
        self.select(cond, if_true, if_false)
    }
    fn select_many_lane(
        &mut self,
        cond: &Self::CellRef,
        true_values: &[Self::CellRef],
        false_values: &[Self::CellRef],
        lane: usize,
    ) -> Result<Self::CellRef, ProvingError> {
        let if_true = true_values
            .get(lane)
            .ok_or_else(|| ProvingError::synthesis("select_many lane out of bounds"))?;
        let if_false = false_values
            .get(lane)
            .ok_or_else(|| ProvingError::synthesis("select_many lane out of bounds"))?;
        self.select(cond, if_true, if_false)
    }
    // ── Casting ───────────────────────────────────────────────────────

    fn int_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn float_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
    fn bool_cast(&mut self, a: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;

    // ── I/O ───────────────────────────────────────────────────────────

    fn read_input(
        &mut self,
        path: &InputPath,
        is_public: bool,
    ) -> Result<Self::CellRef, ProvingError>;
    fn read_external_result(
        &mut self,
        store_idx: u32,
        output_idx: u32,
    ) -> Result<Self::CellRef, ProvingError>;
    fn expose_public(&mut self, a: &Self::CellRef, label: &str) -> Result<(), ProvingError>;
    fn assert_true(&mut self, a: &Self::CellRef) -> Result<(), ProvingError>;
    /// `0 <= idx < hi` discharge. Default impl falls through to the
    /// existing `qge + qlt + and + assert_true` pattern; the Noir backend
    /// overrides to a single `assert((idx as u128) < (hi as u128))`.
    fn assert_u128_lt(
        &mut self,
        idx: &Self::CellRef,
        hi: &Self::CellRef,
    ) -> Result<(), ProvingError> {
        let zero = self.constant_int(0)?;
        let ge = self.gte_i(idx, &zero)?;
        let lt = self.lt_i(idx, hi)?;
        let ok = self.logical_and(&ge, &lt)?;
        self.assert_true(&ok)
    }
    /// Constrain `0 <= idx < hi` and return one low decomposition bit of
    /// `idx`. Backends may cache/fuse calls for the same `(idx, hi, bits)`.
    fn assert_u128_lt_bit(
        &mut self,
        idx: &Self::CellRef,
        hi: u64,
        bit: u32,
        _bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        let hi_cell = self.constant_int(hi as i64)?;
        self.assert_u128_lt(idx, &hi_cell)?;
        let divisor = self.constant_int(1_i64 << bit)?;
        let shifted = self.floor_div_i(idx, &divisor)?;
        let two = self.constant_int(2)?;
        let out = self.mod_i(&shifted, &two)?;
        self.assert_bit(&out)?;
        Ok(out)
    }
    /// Constrain `a` (signed-i64 domain) to be non-negative. Defaults to
    /// the generic comparison; backends with cheap range checks override
    /// (halo2-base: a single 63-bit range check).
    fn assert_non_neg_i(&mut self, a: &Self::CellRef) -> Result<(), ProvingError> {
        let zero = self.constant_int(0)?;
        let ge = self.gte_i(a, &zero)?;
        self.assert_true(&ge)
    }

    /// Constrain `a` to be in {0, 1}. Defaults to `assert_true(eq(a, bool_cast(a)))`
    /// for backends without a dedicated bit gate. The halo2 backend overrides
    /// this with its 1-row `s_bit` gate.
    fn assert_bit(&mut self, a: &Self::CellRef) -> Result<(), ProvingError> {
        let bc = self.bool_cast(a)?;
        let eq = self.eq_i(a, &bc)?;
        self.assert_true(&eq)
    }

    // ── Memory ────────────────────────────────────────────────────────

    fn allocate_memory(
        &mut self,
        segment_id: u32,
        size: u32,
        init: i64,
    ) -> Result<(), ProvingError>;
    fn write_memory(
        &mut self,
        segment_id: u32,
        addr: &Self::CellRef,
        value: &Self::CellRef,
    ) -> Result<(), ProvingError>;
    fn read_memory(
        &mut self,
        segment_id: u32,
        addr: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
    /// `IR::ReadTable`: the slot at `addr`, with no witness for an address
    /// outside the segment; the segment is not written afterwards.
    fn read_table(&mut self, segment_id: u32, addr: &Self::CellRef) -> Result<Self::CellRef, ProvingError>;
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
        max_shape: &[u32],
    ) -> Result<(), ProvingError>;
    fn witness_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        param_name: &str,
        args: &[Self::CellRef],
    ) -> Result<(), ProvingError>;
    /// Return the `(total_length_bound, per_axis_bounds)` recorded for
    /// `array_id` at `allocate_dynamic_ndarray_meta` time. Backends store
    /// these (e.g. in a `dyn_meta_bounds` map keyed by `array_id`) and must
    /// surface them here so the trait-default `assert_dynamic_ndarray_meta`
    /// can range-check the runtime shape against them. The fallback mirrors
    /// the allocation default (`vec![max_length; max_rank]`) for arrays the
    /// backend never recorded.
    fn dyn_meta_axis_bounds(
        &self,
        array_id: u32,
        max_rank: u32,
        max_length: u32,
    ) -> (u32, Vec<u32>);
    /// Constrain a DynamicNDArray's runtime shape against its allocation
    /// envelope: every runtime axis dim is non-negative and `<=` its
    /// per-axis bound, and the product of the dims is `<=` the total length
    /// bound. `args[..max_rank]` are the runtime shape cells; the injection
    /// pass (`DynamicNDArrayMetaAssertInjection`) may append a second block
    /// of `max_rank` constant cells carrying the per-axis bounds, but those
    /// are exactly the values already recorded at allocation time, so this
    /// default reads the bounds from `dyn_meta_axis_bounds` and only uses the
    /// trailing args to validate arity. Provided as a trait default so every
    /// constraint-emitting backend gets the real check; mock overrides it
    /// with a diagnostic (value-inspecting) variant.
    fn assert_dynamic_ndarray_meta(
        &mut self,
        array_id: u32,
        max_rank: u32,
        max_length: u32,
        args: &[Self::CellRef],
    ) -> Result<(), ProvingError> {
        let rank = max_rank as usize;
        if args.len() != rank && args.len() != rank * 2 {
            return Err(ProvingError::synthesis(format!(
                "DynamicNDArray metadata assertion failed for array {}: expected {} runtime shape axes with optional bounds, got {} args",
                array_id, max_rank, args.len()
            )));
        }
        let (total_bound, axis_bounds) =
            self.dyn_meta_axis_bounds(array_id, max_rank, max_length);
        let mut runtime_len = self.constant_int(1)?;
        for (axis, dim) in args[..rank].iter().enumerate() {
            let zero = self.constant_int(0)?;
            let non_negative = self.gte_i(dim, &zero)?;
            self.assert_true(&non_negative)?;
            let axis_bound = axis_bounds.get(axis).copied().unwrap_or(max_length);
            let bound = self.constant_int(axis_bound as i64)?;
            let in_axis_bound = self.lte_i(dim, &bound)?;
            self.assert_true(&in_axis_bound)?;
            runtime_len = self.mul_i(&runtime_len, dim)?;
        }
        let total_bound = self.constant_int(total_bound as i64)?;
        let in_total_bound = self.lte_i(&runtime_len, &total_bound)?;
        self.assert_true(&in_total_bound)?;
        Ok(())
    }
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

    // ── Sort keys ─────────────────────────────────────────────────────

    /// `IR::KeyLess`: `[(v1, i1) < (v2, i2)]` lexicographically. The
    /// default composes the IR's comparisons (`v1 < v2`, or `v1 == v2` and
    /// `i1 < i2`); backends with cheap wide range checks compare the packed
    /// keys `v * 2^index_bits + i` instead.
    fn key_less(
        &mut self,
        v1: &Self::CellRef,
        i1: &Self::CellRef,
        v2: &Self::CellRef,
        i2: &Self::CellRef,
        _index_bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        let value_less = self.lt_i(v1, v2)?;
        let tie = self.eq_i(v1, v2)?;
        let index_less = self.lt_i_bounded(i1, i2)?;
        let tie_less = self.logical_and(&tie, &index_less)?;
        self.logical_or(&value_less, &tie_less)
    }

    /// `IR::PackKey`: `v * 2^index_bits + i`, in field arithmetic.
    fn pack_key(
        &mut self,
        v: &Self::CellRef,
        i: &Self::CellRef,
        index_bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        let scale = self.constant_int(1i64 << index_bits)?;
        let shifted = self.mul_i(v, &scale)?;
        self.add_i(&shifted, i)
    }

    /// `IR::PackedKeyLess`: `[K1 < K2]` for packed keys. The default is the
    /// sign of the field difference (the mock's `lt_i`); backends that
    /// bound their comparisons to the 64-bit window override it.
    fn packed_key_less(
        &mut self,
        k1: &Self::CellRef,
        k2: &Self::CellRef,
        _index_bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        self.lt_i(k1, k2)
    }

    /// `IR::PackedKeyIndex`: the index `i` of a packed key
    /// `K = v * 2^index_bits + i`.
    fn packed_key_index(
        &mut self,
        _key: &Self::CellRef,
        _index_bits: u32,
    ) -> Result<Self::CellRef, ProvingError> {
        Err(ProvingError::synthesis("packed sort keys are not supported by this backend"))
    }

    // ── Prover hints ──────────────────────────────────────────────────

    /// `IR::Hint`: a free value holding lane `rank` of `kind` applied to
    /// `operands`, computed with `kernel::hint_lane` (or, in an emitted
    /// circuit, by the same algorithm in unconstrained code). It must add
    /// no constraint: the IR that uses the hint pins it.
    fn prover_hint(
        &mut self,
        kind: crate::ir_defs::HintKind,
        operands: &[Self::CellRef],
        rank: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;

    // ── Identities checked at verifier challenges ─────────────────────

    /// `IR::IdentityCheck`: assert `identity` among `operands`. The mock
    /// checks it exactly; halo2-base at verifier challenges in its second
    /// phase. A backend without verifier challenges rejects it (the array
    /// lowering never emits it for such a target, `lower::challenge`).
    fn identity_check(
        &mut self,
        identity: crate::ir_defs::Identity,
        _operands: &[Self::CellRef],
    ) -> Result<(), ProvingError> {
        Err(ProvingError::synthesis(format!(
            "identity checks at verifier challenges ({}) are not supported by this backend",
            identity.describe()
        )))
    }

    /// `IR::RescaleRawF`: `floor(raw / 2^P)` of a raw fixed-point sum, with
    /// the rescale checks of `ProductSumF`.
    fn rescale_raw_f(&mut self, _raw: &Self::CellRef) -> Result<Self::CellRef, ProvingError> {
        Err(ProvingError::synthesis("rescaling a raw product sum is not supported by this backend"))
    }

    // ── Hash ──────────────────────────────────────────────────────────

    fn poseidon_hash(&mut self, inputs: &[Self::CellRef]) -> Result<Self::CellRef, ProvingError>;
    fn eq_hash(
        &mut self,
        a: &Self::CellRef,
        b: &Self::CellRef,
    ) -> Result<Self::CellRef, ProvingError>;
}
