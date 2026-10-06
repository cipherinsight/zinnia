//! halo2-base proving backend.
//!
//! This backend uses Axiom's `halo2-base` virtual-region builder rather than
//! Zinnia's hand-recorded halo2 row layout. The goal is to let Zinnia benefit
//! from the same packing strategy used by the handwritten baselines while
//! keeping the semantic oracle in the shared IR interpreter/kernel.

pub mod artifact_format;
pub mod python;
pub mod synthesizer;

use std::sync::Arc;
use std::time::Instant;

use halo2_base::{
    AssignedValue,
    gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        flex_gate::MultiPhaseThreadBreakPoints,
    },
    halo2_proofs::{
        halo2curves::{
            bn256::{Fr as Fp, G1Affine as EqAffine},
            ff::PrimeField,
        },
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey},
        poly::{
            commitment::ParamsProver,
            ipa::{
                commitment::{IPACommitmentScheme, ParamsIPA},
                multiopen::{ProverIPA, VerifierIPA},
                strategy::SingleStrategy,
            },
            VerificationStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    },
};
use halo2_base::gates::circuit::BaseCircuitParams;
use rand::rngs::OsRng;

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

use self::artifact_format::{
    compute_ir_hash, decode as decode_payload, encode as encode_payload,
    Halo2BasePinnedRekeygenV1Payload, HALO2_BASE_PINNED_REKEYGEN_V1,
};
use self::synthesizer::Halo2BaseSynthesizer;

/// k search range for ground-truth parameter estimation.
const ESTIMATE_K_MIN: u32 = 4;
const ESTIMATE_K_MAX: u32 = 22;
/// Rows reserved for blinding factors / poisoned rows when packing cells
/// into columns and when checking row-fit during sizing. Matches the
/// halo2-graph scaffold's MINIMUM_ROWS default (20) so both sides pack
/// columns identically; the real requirement (`meta.minimum_rows()`) is
/// ~10, so 20 is safely conservative.
const UNUSABLE_ROWS: usize = 20;

pub struct Halo2BaseProverBackend;

impl ProverBackend for Halo2BaseProverBackend {
    fn name(&self) -> &'static str {
        "halo2-base"
    }

    fn estimate_params(&self, ir: &IRGraph) -> Result<ProvingParams, ProvingError> {
        // Ground-truth sizing, no tuned thresholds. Two facts drive it:
        //
        // 1. Cell counts are witness-independent, so a dry synthesis into a
        //    virtual builder (the synthesizer substitutes zeros when no
        //    witness is given) yields the exact advice / lookup / fixed cell
        //    counts and the exact public-instance count. Cell counts DO
        //    depend on lookup_bits (range checks decompose into
        //    lookup_bits-wide limbs), so the dry run is cached per
        //    lookup_bits value.
        //
        // 2. Any k whose usable rows hold the instance column and the range
        //    table is viable — `calculate_params` packs cells into as many
        //    columns as needed. But prove time is U-shaped in k: every extra
        //    column is an extra committed polynomial (plus its share of
        //    permutation and lookup arguments), so tiny k explodes the
        //    polynomial count while large k pays 2^k per polynomial. We pick
        //    the k minimizing (committed polynomials) x 2^k, where the
        //    polynomial count is read from halo2's own ConstraintSystem
        //    configured exactly as the real circuit would be.
        use halo2_base::gates::circuit::BaseCircuitParams;
        use halo2_base::halo2_proofs::plonk::ConstraintSystem;
        use std::collections::HashMap;

        let defaults = ProvingParams::default();
        let mut dry_cache: HashMap<u32, CellStats> = HashMap::new();
        let mut dry_run = |lb: u32| -> Result<CellStats, ProvingError> {
            if let Some(v) = dry_cache.get(&lb) {
                return Ok(*v);
            }
            let entry = cell_stats(ir, lb, defaults.precision_bits)?;
            dry_cache.insert(lb, entry);
            Ok(entry)
        };

        let mut best: Option<(u128, u32, u32)> = None; // (cost, k, lookup_bits)
        for k in ESTIMATE_K_MIN..=ESTIMATE_K_MAX {
            let lookup_bits = defaults.lookup_bits.min(k.saturating_sub(1)).max(1);
            let CellStats { advice, lookup, fixed, instances } = dry_run(lookup_bits)?;
            let lookup_used = lookup > 0;

            let usable = (1usize << k).saturating_sub(UNUSABLE_ROWS);
            if usable == 0 || instances > usable {
                continue;
            }
            if lookup_used && (1usize << lookup_bits) > usable {
                continue;
            }

            let num_advice = advice.div_ceil(usable).max(1);
            let num_lookup_advice = if lookup_used { lookup.div_ceil(usable) } else { 0 };
            let num_fixed = fixed.div_ceil(usable).max(1);

            // Configure the real constraint system for this column layout and
            // count the polynomials the prover would commit to.
            let base_params = BaseCircuitParams {
                k: k as usize,
                num_advice_per_phase: vec![num_advice],
                num_fixed,
                num_lookup_advice_per_phase: vec![num_lookup_advice],
                lookup_bits: lookup_used.then_some(lookup_bits as usize),
                num_instance_columns: 1,
            };
            let mut meta = ConstraintSystem::<Fp>::default();
            let _config =
                halo2_base::gates::circuit::BaseConfig::configure(&mut meta, base_params);
            let degree = meta.degree().max(3);
            let perm_chunk = (degree - 2).max(1);
            let perm_columns = meta.permutation().get_columns().len();
            let committed_polys = num_advice
                + num_lookup_advice
                + meta.lookups().len() * 2 // mv_lookup: multiplicity + running-sum poly
                + perm_columns.div_ceil(perm_chunk) // permutation grand products
                + (degree - 1) // quotient pieces
                + 1; // vanishing blinder
            let cost = (committed_polys as u128) << k;
            if best.map_or(true, |(c, _, _)| cost < c) {
                best = Some((cost, k, lookup_bits));
            }
        }

        match best {
            Some((_, k, lookup_bits)) => Ok(ProvingParams {
                k,
                lookup_bits,
                ..defaults
            }),
            None => Err(ProvingError::other(format!(
                "no k in [{}..{}] fits the circuit's public instances",
                ESTIMATE_K_MIN, ESTIMATE_K_MAX
            ))),
        }
    }

    fn prove(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> Result<ProofArtifact, ProvingError> {
        let ir_stmts = ir.export_stmts();
        let ir_hash = compute_ir_hash(&ir_stmts)
            .map_err(|e| ProvingError::other(format!("ir_hash compute failed: {}", e)))?;
        self.prove_with_cached_artifact(ir, witness, params, &ir_stmts, &ir_hash)
    }

    fn prove_with_cached_artifact(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
        ir_stmts: &[serde_json::Value],
        ir_hash: &str,
    ) -> Result<ProofArtifact, ProvingError> {
        let canonical_ir_hash = compute_ir_hash(ir_stmts)
            .map_err(|e| ProvingError::other(format!("ir_hash compute failed: {}", e)))?;
        let ir_hash = if ir_hash == canonical_ir_hash {
            ir_hash
        } else {
            canonical_ir_hash.as_str()
        };

        let ipa_params = Arc::new(ParamsIPA::<EqAffine>::new(params.k));

        let (mut keygen_circuit, _keygen_public) =
            build_circuit(ir, Some(witness), params, CircuitBuilderStage::Keygen, None)?;
        let vk = without_panic("keygen_vk", params.k, || keygen_vk(&*ipa_params, &keygen_circuit))?
            .map_err(|e| ProvingError::ProvingFailed {
                detail: format!("halo2-base keygen_vk failed: {:?}", e),
            })?;
        let pk = without_panic("keygen_pk", params.k, || keygen_pk(&*ipa_params, vk, &keygen_circuit))?
            .map_err(|e| ProvingError::ProvingFailed {
                detail: format!("halo2-base keygen_pk failed: {:?}", e),
            })?;
        let base_params = keygen_circuit.params();
        let break_points = keygen_circuit.break_points();
        keygen_circuit.clear();

        let pinning = (base_params.clone(), break_points.clone());
        let (prover_circuit, public_inputs) =
            build_circuit(ir, Some(witness), params, CircuitBuilderStage::Prover, Some(pinning))?;
        let instances = vec![public_inputs.clone()];
        let instance_refs: Vec<&[Fp]> = instances.iter().map(|v| v.as_slice()).collect();

        let t_create = Instant::now();
        let mut transcript: Blake2bWrite<Vec<u8>, EqAffine, Challenge255<_>> =
            TranscriptWriterBuffer::<Vec<u8>, EqAffine, _>::init(Vec::new());
        create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
            &ipa_params,
            &pk,
            &[prover_circuit],
            &[&instance_refs],
            OsRng,
            &mut transcript,
        )
        .map_err(|e| ProvingError::ProvingFailed {
            detail: format!("halo2-base create_proof failed: {:?}", e),
        })?;
        let proof_bytes = transcript.finalize();
        let prove_seconds = t_create.elapsed().as_secs_f64();

        let payload = Halo2BasePinnedRekeygenV1Payload {
            ir_stmts: ir_stmts.to_vec(),
            precision_bits: params.precision_bits,
            lookup_bits: params.lookup_bits,
            base_params,
            break_points,
            provenance: ir
                .provenance
                .to_wire(&|vid| ir.value_to_stmt.get(&vid).copied()),
        };
        let vk_bytes = encode_payload(&payload).map_err(|e| {
            ProvingError::other(format!(
                "vk_bytes (halo2-base-pinned-rekeygen-v1) encode failed: {}",
                e
            ))
        })?;

        Ok(ProofArtifact {
            backend: "halo2-base".to_string(),
            format: HALO2_BASE_PINNED_REKEYGEN_V1.to_string(),
            vk_bytes,
            ir_hash: ir_hash.to_string(),
            proof_bytes: hex_encode(&proof_bytes),
            public_values: public_inputs.iter().map(encode_fp).collect(),
            k: params.k,
            prove_seconds,
        })
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        if artifact.format != HALO2_BASE_PINNED_REKEYGEN_V1 {
            return Err(ProvingError::VerificationFailed {
                detail: format!(
                    "Unsupported halo2-base artifact format: {:?}",
                    artifact.format
                ),
            });
        }
        let proof_bytes =
            hex_decode(&artifact.proof_bytes).map_err(|e| ProvingError::VerificationFailed {
                detail: format!("Invalid proof hex: {}", e),
            })?;
        let public_inputs: Vec<Fp> = artifact
            .public_values
            .iter()
            .map(|s| decode_fp(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ProvingError::VerificationFailed {
                detail: format!("Invalid public-input encoding: {}", e),
            })?;
        let payload =
            decode_payload(&artifact.vk_bytes).map_err(|e| ProvingError::VerificationFailed {
                detail: format!("Failed to decode halo2-base payload: {}", e),
            })?;

        // ── IR-hash binding check (before re-keygen / proof verify) ────
        //
        // Recompute the IR hash from the shipped IR and require it to match
        // the artifact's recorded `ir_hash`. A mismatch means the verifier
        // is being handed a proof for a different circuit than the IR it is
        // told it is checking. Surface the typed `IrHashMismatch` (not a
        // generic invalid result) so artifact-transport regressions are
        // attributable. Mirrors the legacy halo2-IPA verify path.
        let computed_ir_hash =
            compute_ir_hash(&payload.ir_stmts).map_err(|e| ProvingError::VerificationFailed {
                detail: format!("ir_hash recompute failed: {}", e),
            })?;
        if computed_ir_hash != artifact.ir_hash {
            return Err(ProvingError::ir_hash_mismatch(
                artifact.ir_hash.clone(),
                computed_ir_hash,
            ));
        }

        let mut graph = IRGraph::import_stmts(&payload.ir_stmts).map_err(|e| {
            ProvingError::VerificationFailed {
                detail: format!("IR import failed: {}", e),
            }
        })?;
        // Re-attach the provenance records — synthesis consumes them
        // (Step 2: interval/affine narrowing changes the emitted gate
        // count), so the verify-side rebuild must see exactly what the
        // prover saw. Stmt-keyed wire form: import_stmts mints fresh
        // ValueIds, so the table is re-keyed through stmt ids.
        let vid_of_stmt: std::collections::HashMap<_, _> = graph
            .stmts
            .iter()
            .map(|st| (st.stmt_id, st.value_id))
            .collect();
        graph.provenance = crate::provenance::ProvenanceTable::from_wire(
            &payload.provenance,
            &|sid| vid_of_stmt.get(&sid).copied(),
        );
        let recovered_params = ProvingParams {
            k: artifact.k,
            precision_bits: payload.precision_bits,
            lookup_bits: payload.lookup_bits,
        };
        let keygen_circuit = build_circuit(
            &graph,
            None,
            &recovered_params,
            CircuitBuilderStage::Keygen,
            Some((payload.base_params, payload.break_points)),
        )?
        .0;
        let ipa_params = ParamsIPA::<EqAffine>::new(artifact.k);
        let vk = keygen_vk(&ipa_params, &keygen_circuit).map_err(|e| {
            ProvingError::VerificationFailed {
                detail: format!("halo2-base re-keygen failed: {:?}", e),
            }
        })?;

        let mut transcript =
            Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof_bytes.as_slice());
        let strategy = SingleStrategy::new(&ipa_params);
        match verify_proof::<
            IPACommitmentScheme<EqAffine>,
            VerifierIPA<EqAffine>,
            _,
            _,
            SingleStrategy<EqAffine>,
        >(
            &ipa_params,
            &vk,
            strategy,
            &[&[public_inputs.as_slice()]],
            &mut transcript,
        ) {
            Ok(_) => Ok(VerifyResult::ok()),
            Err(e) => Ok(VerifyResult::invalid(format!(
                "halo2-base verify_proof failed: {:?}",
                e
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Separate key generation, proving and verification
//
// `prove` above regenerates the IPA parameters and runs keygen with the
// prover's witness on every call, and `verify` re-runs keygen from the IR
// shipped in the artifact. That couples one-time setup with every proof and
// every verification. The three functions below split the steps the way a
// deployment (and the handwritten halo2 baseline) does: keys once, without a
// witness; then proofs; then verification against a key the verifier holds.
// ---------------------------------------------------------------------------

/// Keys for one compiled circuit and one parameter choice, generated
/// without a witness (exactly the synthesis a verifier would run).
pub struct Halo2BaseKeys {
    pub params: ProvingParams,
    pub ipa_params: ParamsIPA<EqAffine>,
    pub pk: ProvingKey<EqAffine>,
    /// Column layout and thread break points of the keygen synthesis; the
    /// prover's synthesis is pinned to them so it matches the keys.
    pub base_params: BaseCircuitParams,
    pub break_points: MultiPhaseThreadBreakPoints,
    /// `prove::ir_hash::compute_ir_hash` of the circuit the keys were made
    /// for (the hash `CompiledIR` caches).
    pub ir_hash: String,
}

impl Halo2BaseKeys {
    pub fn vk(&self) -> &VerifyingKey<EqAffine> {
        self.pk.get_vk()
    }

    /// The verifying key in halo2's raw-bytes encoding (for size reports
    /// and out-of-process verifiers).
    pub fn vk_bytes(&self) -> Vec<u8> {
        self.vk().to_bytes(halo2_base::halo2_proofs::SerdeFormat::RawBytes)
    }
}

/// A proof made with [`prove_with_keys`], with its two timed phases.
pub struct TimedProof {
    pub proof: Vec<u8>,
    pub public_values: Vec<Fp>,
    /// Witness synthesis: the IR walk filling the pinned circuit builder.
    pub witness_seconds: f64,
    /// `create_proof` plus transcript finalisation.
    pub prove_seconds: f64,
}

/// Generate IPA parameters and proving/verifying keys for `ir` at `params`
/// without a witness. The circuit shape is witness-independent (tested by
/// `structural_fingerprint`), so these keys serve every input.
pub fn keygen(ir: &IRGraph, params: &ProvingParams) -> Result<Halo2BaseKeys, ProvingError> {
    let ir_hash = crate::prove::ir_hash::compute_ir_hash(&ir.export_stmts())
        .map_err(|e| ProvingError::other(format!("ir_hash compute failed: {}", e)))?;
    let ipa_params = ParamsIPA::<EqAffine>::new(params.k);
    let (mut circuit, _) = build_circuit(ir, None, params, CircuitBuilderStage::Keygen, None)?;
    let vk = without_panic("keygen_vk", params.k, || keygen_vk(&ipa_params, &circuit))?.map_err(|e| {
        ProvingError::ProvingFailed { detail: format!("halo2-base keygen_vk failed: {:?}", e) }
    })?;
    let pk = without_panic("keygen_pk", params.k, || keygen_pk(&ipa_params, vk, &circuit))?.map_err(|e| {
        ProvingError::ProvingFailed { detail: format!("halo2-base keygen_pk failed: {:?}", e) }
    })?;
    let base_params = circuit.params();
    let break_points = circuit.break_points();
    circuit.clear();
    Ok(Halo2BaseKeys { params: params.clone(), ipa_params, pk, base_params, break_points, ir_hash })
}

/// [`keygen`] at `params` (from `estimate_params`); should the layout still
/// fail at that `k` (an error naming it), the next `k` is tried, up to the
/// estimate's upper bound.
pub fn keygen_or_next_k(ir: &IRGraph, params: &ProvingParams) -> Result<Halo2BaseKeys, ProvingError> {
    let mut params = params.clone();
    loop {
        match keygen(ir, &params) {
            Err(ProvingError::ProvingFailed { detail }) if detail.contains("failed at k=") && params.k < ESTIMATE_K_MAX => {
                params.k += 1;
                params.lookup_bits = ProvingParams::default().lookup_bits.min(params.k - 1);
            }
            other => return other,
        }
    }
}

/// Prove `ir` on `witness` with keys from [`keygen`] (no parameter
/// generation, no keygen).
pub fn prove_with_keys(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    keys: &Halo2BaseKeys,
) -> Result<TimedProof, ProvingError> {
    let t_witness = Instant::now();
    let pinning = (keys.base_params.clone(), keys.break_points.clone());
    let (circuit, public_values) =
        build_circuit(ir, Some(witness), &keys.params, CircuitBuilderStage::Prover, Some(pinning))?;
    let witness_seconds = t_witness.elapsed().as_secs_f64();

    let t_prove = Instant::now();
    let instance_refs: Vec<&[Fp]> = vec![public_values.as_slice()];
    let mut transcript: Blake2bWrite<Vec<u8>, EqAffine, Challenge255<_>> =
        TranscriptWriterBuffer::<Vec<u8>, EqAffine, _>::init(Vec::new());
    create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
        &keys.ipa_params,
        &keys.pk,
        &[circuit],
        &[&instance_refs],
        OsRng,
        &mut transcript,
    )
    .map_err(|e| ProvingError::ProvingFailed {
        detail: format!("halo2-base create_proof failed: {:?}", e),
    })?;
    let proof = transcript.finalize();
    let prove_seconds = t_prove.elapsed().as_secs_f64();
    Ok(TimedProof { proof, public_values, witness_seconds, prove_seconds })
}

/// Verify `proof` for `public_values` against the verifying key in `keys`
/// (no keygen, no IR).
pub fn verify_with_keys(keys: &Halo2BaseKeys, proof: &[u8], public_values: &[Fp]) -> VerifyResult {
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);
    let strategy = SingleStrategy::new(&keys.ipa_params);
    match verify_proof::<IPACommitmentScheme<EqAffine>, VerifierIPA<EqAffine>, _, _, SingleStrategy<EqAffine>>(
        &keys.ipa_params,
        keys.vk(),
        strategy,
        &[&[public_values]],
        &mut transcript,
    ) {
        Ok(_) => VerifyResult::ok(),
        Err(e) => VerifyResult::invalid(format!("halo2-base verify_proof failed: {:?}", e)),
    }
}

/// Cell counts of one synthesized circuit (witness-independent).
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct CellStats {
    pub advice: usize,
    pub lookup: usize,
    pub fixed: usize,
    pub instances: usize,
}

/// The cell counts of `ir` synthesized without a witness at `lookup_bits`.
/// Counts depend on the lookup width (range checks decompose into limbs of
/// that width) but not on `k`.
pub fn cell_stats(ir: &IRGraph, lookup_bits: u32, precision_bits: u32) -> Result<CellStats, ProvingError> {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Keygen);
    builder.set_k(ESTIMATE_K_MAX as usize);
    builder.set_lookup_bits(lookup_bits as usize);
    builder.set_instance_columns(1);
    let params = ProvingParams { k: ESTIMATE_K_MAX, lookup_bits, precision_bits };
    let instances = {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params, &ir.provenance);
        synth.plan_ahead(ir);
        interpret_ir(ir, &mut synth)?;
        synth.public_values().len()
    };
    let stats = builder.statistics();
    let cells = CellStats {
        advice: stats.gate.total_advice_per_phase.iter().sum(),
        lookup: stats.total_lookup_advice_per_phase.iter().sum(),
        fixed: stats.gate.total_fixed,
        instances,
    };
    builder.clear();
    Ok(cells)
}

/// Advice and lookup cells per IR statement, `(statement id, advice,
/// lookup)`, for a witness-free synthesis at `lookup_bits` (a measurement
/// aid: it shows which statements a circuit's cells come from).
pub fn cell_profile(
    ir: &IRGraph,
    lookup_bits: u32,
    precision_bits: u32,
) -> Result<Vec<(u32, usize, usize)>, ProvingError> {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Keygen);
    builder.set_k(ESTIMATE_K_MAX as usize);
    builder.set_lookup_bits(lookup_bits as usize);
    builder.set_instance_columns(1);
    let params = ProvingParams { k: ESTIMATE_K_MAX, lookup_bits, precision_bits };
    let profile = {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params, &ir.provenance);
        synth.plan_ahead(ir);
        synth.enable_profile();
        interpret_ir(ir, &mut synth)?;
        synth.take_profile().map(|p| p.per_stmt).unwrap_or_default()
    };
    builder.clear();
    Ok(profile)
}

/// Hex encoding of a field element, as in `ProofArtifact::public_values`.
pub fn fp_to_hex(v: &Fp) -> String {
    encode_fp(v)
}

/// Inverse of [`fp_to_hex`].
pub fn fp_from_hex(s: &str) -> Result<Fp, String> {
    decode_fp(s)
}

/// Keygen-stage structural fingerprint for the witness-independence
/// regression test (docs/provenance-consolidation-plan.md, Step 0a):
/// synthesizes the circuit with the given witness and renders the cell
/// statistics. A sound (witness-independent) synthesizer yields the same
/// fingerprint for every witness of the same compiled circuit; any drift
/// means emission decisions leaked witness data.
pub fn structural_fingerprint(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    params: &ProvingParams,
) -> Result<String, ProvingError> {
    fingerprint_with_witness(ir, Some(witness), params)
}

/// Witness-LESS keygen fingerprint — exactly the synthesis the verifier
/// re-runs (all reads default to zero). A sound synthesizer yields the
/// same structural fingerprint here as for any concrete witness; a
/// mismatch with [`structural_fingerprint`] means the prover-side circuit
/// shape leaked the witness and the resulting proof would fail verify-side
/// re-keygen. The two-witness test asserts
/// `Some(a) == Some(b) == None`.
pub fn structural_fingerprint_witnessless(
    ir: &IRGraph,
    params: &ProvingParams,
) -> Result<String, ProvingError> {
    fingerprint_with_witness(ir, None, params)
}

fn fingerprint_with_witness(
    ir: &IRGraph,
    witness: Option<&ResolvedWitness>,
    params: &ProvingParams,
) -> Result<String, ProvingError> {
    let (builder, public_values) =
        build_circuit(ir, witness, params, CircuitBuilderStage::Keygen, None)?;
    let stats = builder.statistics();
    Ok(format!(
        "advice={:?} lookup={:?} fixed={} instances={}",
        stats.gate.total_advice_per_phase,
        stats.total_lookup_advice_per_phase,
        stats.gate.total_fixed,
        public_values.len(),
    ))
}

fn build_circuit(
    ir: &IRGraph,
    witness: Option<&ResolvedWitness>,
    params: &ProvingParams,
    stage: CircuitBuilderStage,
    pinning: Option<(halo2_base::gates::circuit::BaseCircuitParams, MultiPhaseThreadBreakPoints)>,
) -> Result<(BaseCircuitBuilder<Fp>, Vec<Fp>), ProvingError> {
    let mut builder = match (stage, pinning) {
        (CircuitBuilderStage::Prover, Some((base_params, break_points))) => {
            BaseCircuitBuilder::prover(base_params, break_points)
        }
        (_, Some((base_params, break_points))) => {
            let mut builder = BaseCircuitBuilder::from_stage(stage);
            builder.set_params(base_params);
            builder.set_break_points(break_points);
            builder
        }
        (_, None) => {
            let mut builder = BaseCircuitBuilder::from_stage(stage);
            builder.set_k(params.k as usize);
            builder.set_lookup_bits(params.lookup_bits as usize);
            builder.set_instance_columns(1);
            builder
        }
    };
    if builder.assigned_instances.is_empty() {
        builder.set_instance_columns(1);
    }

    let public_values = if let Some(public_values) =
        try_build_public_layout_only(&mut builder, ir, witness)?
    {
        public_values
    } else {
        let mut synth =
            Halo2BaseSynthesizer::new(&mut builder, witness.cloned(), params.clone(), &ir.provenance);
        synth.plan_ahead(ir);
        interpret_ir(ir, &mut synth).map_err(|e| {
            ProvingError::synthesis(format!("halo2-base synthesis failed: {}", e))
        })?;
        // Always-on witness-satisfaction check: halo2's `create_proof` does
        // NOT verify the witness satisfies the gates (it emits a proof that
        // `verify_proof` later rejects). The synthesizer records, during
        // this same walk, the first assert/bit gate whose concrete witness
        // violates the gate; surface it so an unsatisfiable witness (e.g.
        // `assert(False)`) fails fast rather than producing a valid-looking
        // artifact. Fires only when a witness is bound (prove-side keygen +
        // prover stages), never on the witness-less verify re-keygen.
        if let Some(detail) = synth.first_unsat() {
            return Err(ProvingError::ProvingFailed {
                detail: detail.to_string(),
            });
        }
        synth.public_values().to_vec()
    };

    if stage != CircuitBuilderStage::Prover {
        calculate_layout(&mut builder);
        if std::env::var("ZINNIA_HALO2_BASE_DEBUG_CELLS").is_ok() {
            let stats = builder.statistics();
            eprintln!(
                "[halo2-base cells] advice={:?} lookup={:?} fixed={} instances={}",
                stats.gate.total_advice_per_phase,
                stats.total_lookup_advice_per_phase,
                stats.gate.total_fixed,
                public_values.len(),
            );
        }
    }
    Ok((builder, public_values))
}

/// The column layout for the builder's cells at its `k`, reserving
/// `UNUSABLE_ROWS` rows per column.
///
/// halo2-base's `calculate_params` reserves those rows for advice and lookup
/// columns but sizes the constants' fixed columns as `ceil(fixed / 2^k)`,
/// although the constants cannot use the blinding rows either: a circuit
/// with many constants then fails to lay out (`NotEnoughRowsAvailable` at
/// key generation). The fixed columns are sized like the others here, which
/// is also what `estimate_params`' cost model assumes.
fn calculate_layout(builder: &mut BaseCircuitBuilder<Fp>) {
    let mut params = builder.calculate_params(Some(UNUSABLE_ROWS));
    let usable = (1usize << params.k).saturating_sub(UNUSABLE_ROWS).max(1);
    let fixed = builder.statistics().gate.total_fixed;
    params.num_fixed = fixed.div_ceil(usable).max(params.num_fixed);
    builder.set_params(params);
}

fn try_build_public_layout_only(
    builder: &mut BaseCircuitBuilder<Fp>,
    ir: &IRGraph,
    witness: Option<&ResolvedWitness>,
) -> Result<Option<Vec<Fp>>, ProvingError> {
    let mut reads: Vec<Option<(AssignedValue<Fp>, Fp)>> = vec![None; ir.stmts.len()];
    let mut public_values = Vec::new();
    // Buffer the instance cells locally rather than pushing them into the
    // shared `builder.assigned_instances` as we go. If this fast path bails
    // out partway (returns `None`), `build_circuit` re-runs the full
    // synthesizer on the SAME builder; any instance cells leaked here would
    // survive and be appended to the full-synthesis instances, leaving
    // `assigned_instances` longer than the public-input vector and breaking
    // the instance copy constraints (C17). We only commit on success.
    let mut pending_instances: Vec<AssignedValue<Fp>> = Vec::new();

    for stmt in &ir.stmts {
        match &stmt.ir {
            IR::ReadInteger { path, is_public } | IR::ReadFloat { path, is_public } => {
                let value = witness
                    .and_then(|w| w.resolve_path(path).ok())
                    .unwrap_or_else(Fp::zero);
                let assigned = builder.main(0).load_witness(value);
                let idx = stmt.stmt_id as usize;
                if idx >= reads.len() {
                    return Ok(None);
                }
                reads[idx] = Some((assigned, value));
                if *is_public {
                    pending_instances.push(assigned);
                    public_values.push(value);
                }
            }
            IR::ExposePublicI | IR::ExposePublicF => {
                if stmt.arguments.len() != 1 {
                    return Ok(None);
                }
                let arg = stmt.arguments[0] as usize;
                let Some((assigned, value)) = reads.get(arg).and_then(|v| *v) else {
                    return Ok(None);
                };
                pending_instances.push(assigned);
                public_values.push(value);
            }
            IR::AllocateMemory { .. }
            | IR::StructuralPredicate { .. }
            | IR::ScalarPrecondition { .. } => {}
            _ => return Ok(None),
        }
    }

    if public_values.is_empty() {
        return Ok(None);
    }
    // Success: commit the buffered instance cells exactly once.
    builder.assigned_instances[0].extend(pending_instances);
    Ok(Some(public_values))
}

/// Run a halo2 step that lays out the circuit, turning a halo2-base panic
/// (for example `NotEnoughRowsAvailable` when `k` is too small for the
/// fixed cells) into an error instead of unwinding into Python.
fn without_panic<T>(step: &str, k: u32, f: impl FnOnce() -> T) -> Result<T, ProvingError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        let reason = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".to_string());
        ProvingError::ProvingFailed { detail: format!("halo2-base {} failed at k={}: {}", step, k, reason) }
    })
}

fn encode_fp(v: &Fp) -> String {
    let repr = v.to_repr();
    repr.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
}

fn decode_fp(s: &str) -> Result<Fp, String> {
    let bytes = hex_decode(s)?;
    if bytes.len() != 32 {
        return Err(format!("Expected 32-byte Fp, got {} bytes", bytes.len()));
    }
    let mut repr = <Fp as PrimeField>::Repr::default();
    repr.as_mut().copy_from_slice(&bytes);
    Option::<Fp>::from(Fp::from_repr(repr))
        .ok_or_else(|| "Invalid Fp encoding (non-canonical)".to_string())
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
