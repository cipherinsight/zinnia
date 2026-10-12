//! halo2-base proving backend.
//!
//! This backend uses Axiom's `halo2-base` virtual-region builder rather than
//! Zinnia's hand-recorded halo2 row layout. The goal is to let Zinnia benefit
//! from the same packing strategy used by the handwritten baselines while
//! keeping the semantic oracle in the shared IR interpreter/kernel.

pub mod artifact_format;
pub mod circuit;
mod identities;
pub mod keys;
pub mod python;
pub mod synthesizer;

#[cfg(test)]
mod parameter_tests;

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
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
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
use rand::rngs::OsRng;

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

use self::circuit::{CircuitParams, Halo2BaseCircuit};
use self::artifact_format::{
    compute_ir_hash, decode as decode_payload, encode as encode_payload,
    Halo2BasePinnedRekeygenV1Payload, HALO2_BASE_PINNED_REKEYGEN_V1,
};
use self::synthesizer::Halo2BaseSynthesizer;

/// k search range for ground-truth parameter estimation.
const ESTIMATE_K_MIN: u32 = 4;
const ESTIMATE_K_MAX: u32 = 22;
/// How far above the cheapest committed-polynomials x 2^k estimate a layout may
/// be and still win on cell count. The estimate is this inaccurate: on
/// ds1000_case0436 the layout it rates 4% more expensive proves faster.
const WORK_SLACK_PERCENT: u128 = 10;
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
        // Ground-truth sizing. Two facts drive it:
        //
        // 1. Cell counts are witness-independent, so a dry synthesis into a
        //    virtual builder (the synthesizer substitutes zeros when no
        //    witness is given) yields the exact advice / lookup / fixed cell
        //    counts and the exact public-instance count. Cell counts DO
        //    depend on lookup_bits (range checks decompose into
        //    lookup_bits-wide limbs), so the dry run is cached per
        //    lookup_bits value.
        //
        // 2. Prove time is U-shaped in k: every extra column is an extra
        //    committed polynomial, while large k pays 2^k per polynomial. The
        //    committed-polynomials x 2^k proxy estimates that work, but only
        //    roughly: layouts it rates a few percent apart can prove in the
        //    reverse order. So among the layouts within WORK_SLACK_PERCENT of
        //    the cheapest estimate, take the fewest advice + lookup cells
        //    (including second-phase, table and RAM advice, exactly as in
        //    cell_stats), then the cheaper estimate, k and lookup width for
        //    deterministic ties. Outside the slack a smaller circuit is not
        //    worth it: a tiny circuit with an 8-bit lookup table has fewer
        //    cells but needs k = 9 for the table, and proves several times
        //    slower than at the k its cells need. Every width up to the
        //    existing maximum is searched (limb costs need not decrease
        //    monotonically with width); all candidates must pass
        //    committed_polys' row-fit checks.
        use std::collections::HashMap;

        let defaults = ProvingParams::default();
        let mut dry_cache: HashMap<u32, CellStats> = HashMap::new();
        let mut dry_run = |lb: u32| -> Result<CellStats, ProvingError> {
            if let Some(v) = dry_cache.get(&lb) {
                return Ok(v.clone());
            }
            let entry = cell_stats(ir, lb, defaults.precision_bits)?;
            dry_cache.insert(lb, entry.clone());
            Ok(entry)
        };

        // (cells, work estimate, k, lookup_bits) of every layout that fits.
        let mut candidates: Vec<(usize, u128, u32, u32)> = Vec::new();
        for k in ESTIMATE_K_MIN..=ESTIMATE_K_MAX {
            for lookup_bits in 1..=defaults.lookup_bits.min(k.saturating_sub(1)) {
                let stats = dry_run(lookup_bits)?;
                let Some(polys) = committed_polys(&stats, k, lookup_bits) else { continue };
                candidates.push((stats.advice + stats.lookup, (polys as u128) << k, k, lookup_bits));
            }
        }
        let cheapest = candidates.iter().map(|c| c.1).min();
        let best = cheapest.and_then(|cheapest| {
            candidates
                .into_iter()
                .filter(|c| c.1 * 100 <= cheapest * (100 + WORK_SLACK_PERCENT))
                .min()
        });

        match best {
            Some((_, _, k, lookup_bits)) => Ok(ProvingParams {
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
        let circuit_params = keygen_circuit.params();
        let break_points = keygen_circuit.break_points();
        keygen_circuit.builder.clear();

        let pinning = (circuit_params.clone(), break_points.clone());
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
            base_params: circuit_params.base,
            break_points,
            challenges: circuit_params.challenges,
            table_layout: circuit_params.table_layout.clone(),
            ram_traces: circuit_params.ram_traces,
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
        // attributable.
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
            Some((
                CircuitParams {
                    base: payload.base_params,
                    challenges: payload.challenges,
                    table_layout: payload.table_layout,
                    ram_traces: payload.ram_traces,
                },
                payload.break_points,
            )),
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

/// The polynomials the prover commits to for a circuit with `stats` at `k`
/// and `lookup_bits` (the secondary objective in `estimate_params`),
/// or `None` when the circuit does not fit `k`.
pub fn committed_polys(stats: &CellStats, k: u32, lookup_bits: u32) -> Option<usize> {
    use halo2_base::gates::circuit::BaseCircuitParams;
    use halo2_base::halo2_proofs::plonk::ConstraintSystem;
    let CellStats {
        advice, second_advice, challenges, table_advice, table_rows, tables, ram_advice, ram_rows, ram_traces, lookup, fixed, instances,
    } = stats;
    let lookup_used = *lookup > 0;
    let usable = (1usize << k).saturating_sub(UNUSABLE_ROWS);
    if usable == 0 || *instances > usable || (lookup_used && (1usize << lookup_bits) > usable) || *table_rows > usable || *ram_rows > usable {
        return None;
    }
    let num_advice = (advice - second_advice - table_advice - ram_advice).div_ceil(usable).max(1);
    // The RAM traces (`circuit/ram.rs`): per trace 8 advice columns, a
    // shuffle (one grand product) and the gap lookup.
    let ram_polys = ram_traces * (8 + 1 + 2);
    // The memory tables (`circuit/tables.rs`): per table group the value
    // column, two advice columns per key set, one lookup argument per key set.
    let (groups, _) = circuit::tables::pack(tables, usable);
    let table_polys: usize = circuit::tables::key_sets(&groups, usable).iter().map(|sets| 1 + 2 * sets + 2 * sets).sum();
    // The second phase (challenge-based checks, `circuit.rs`): its gate
    // columns and the challenges' column.
    let second_columns = if *challenges > 0 { second_advice.div_ceil(usable).max(1) + 1 } else { 0 };
    let num_lookup_advice = if lookup_used { lookup.div_ceil(usable) } else { 0 };
    let num_fixed = fixed.div_ceil(usable).max(1);

    // Configure the real constraint system for this column layout and count
    // the polynomials the prover would commit to.
    let base_params = BaseCircuitParams {
        k: k as usize,
        num_advice_per_phase: vec![num_advice],
        num_fixed,
        num_lookup_advice_per_phase: vec![num_lookup_advice],
        lookup_bits: lookup_used.then_some(lookup_bits as usize),
        num_instance_columns: 1,
    };
    let mut meta = ConstraintSystem::<Fp>::default();
    let _config = halo2_base::gates::circuit::BaseConfig::configure(&mut meta, base_params);
    let degree = meta.degree().max(3);
    let perm_chunk = (degree - 2).max(1);
    let perm_columns = meta.permutation().get_columns().len();
    Some(
        num_advice
            + num_lookup_advice
            + second_columns
            + table_polys
            + ram_polys
            + meta.lookups().len() * 2 // mv_lookup: multiplicity + running-sum poly
            + perm_columns.div_ceil(perm_chunk) // permutation grand products
            + (degree - 1) // quotient pieces
            + 1, // vanishing blinder
    )
}

/// The polynomials the prover commits to for `ir` at `params`, counted on
/// the constraint system the real circuit configures (every phase, the
/// challenge column, the memory tables and RAM traces): its advice columns,
/// three per lookup argument (halo2's permuted input, permuted table and
/// product), one product per shuffle, the permutation argument's products
/// (`degree - 2` columns each), and the quotient pieces and blinder. A
/// measurement aid for `committed_polys`, which estimates the same from cell
/// counts without a synthesis of the circuit. Returns JSON with the parts.
pub fn configured_polys(ir: &IRGraph, params: &ProvingParams) -> Result<serde_json::Value, ProvingError> {
    use halo2_base::halo2_proofs::plonk::{Circuit, ConstraintSystem};
    let (mut circuit, _) = build_circuit(ir, None, params, CircuitBuilderStage::Keygen, None)?;
    let mut meta = ConstraintSystem::<Fp>::default();
    let _config = Halo2BaseCircuit::configure_with_params(&mut meta, circuit.params());
    circuit.builder.clear();
    let degree = meta.degree().max(3);
    let (advice, lookups, shuffles) = (meta.num_advice_columns(), meta.lookups().len(), meta.shuffles().len());
    let permutation = meta.permutation().get_columns().len();
    let permutation_products = permutation.div_ceil(degree - 2);
    let polys = advice + 3 * lookups + shuffles + permutation_products + (degree - 1) + 1;
    Ok(serde_json::json!({
        "k": params.k, "degree": degree, "advice_columns": advice, "lookups": lookups, "shuffles": shuffles,
        "permutation_columns": permutation, "committed_polys": polys, "poly_cost": (polys as u64) << params.k,
    }))
}

/// Cell counts of one synthesized circuit (witness-independent).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CellStats {
    /// All advice cells, including second-phase, table and RAM advice.
    pub advice: usize,
    /// Of those, the second phase's: its checks' cells and the challenge
    /// cells (`circuit.rs`).
    pub second_advice: usize,
    /// Verifier challenges the circuit declares (0: one phase).
    pub challenges: usize,
    /// Of `advice`, the memory tables' cells (`circuit/tables.rs`), the rows
    /// their table column needs and the reads looked up in them.
    pub table_advice: usize,
    pub table_rows: usize,
    /// `(rows, reads)` of each table, in tag order.
    pub tables: Vec<(usize, usize)>,
    /// Of `advice`, the RAM traces' cells (`circuit/ram.rs`), the rows of
    /// the longest trace and the number of traces.
    pub ram_advice: usize,
    pub ram_rows: usize,
    pub ram_traces: usize,
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
    let (instances, second, tables, ram) = if let Some(public_values) =
        try_build_public_layout_only(&mut builder, ir, None)?
    {
        (
            public_values.len(),
            circuit::SecondPhase::default(),
            circuit::Tables::default(),
            circuit::Ram::default(),
        )
    } else {
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params);
        synth.plan_ahead(ir);
        interpret_ir(ir, &mut synth)?;
        let parts = (std::mem::take(&mut synth.second_phase), std::mem::take(&mut synth.tables), std::mem::take(&mut synth.ram));
        (synth.public_values().len(), parts.0, parts.1, parts.2)
    };
    let stats = builder.statistics();
    let shape = second.shape();
    let second_advice = shape.advice + second.challenges();
    let cells = CellStats {
        advice: stats.gate.total_advice_per_phase.iter().sum::<usize>() + second_advice + tables.advice_cells() + ram.advice_cells(),
        ram_advice: ram.advice_cells(),
        ram_rows: ram.rows_needed(),
        ram_traces: ram.traces.len(),
        table_advice: tables.advice_cells(),
        table_rows: tables.rows_needed(),
        tables: tables.shape(),
        second_advice,
        challenges: second.challenges(),
        lookup: stats.total_lookup_advice_per_phase.iter().sum(),
        fixed: stats.gate.total_fixed + shape.constants,
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
        let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, params);
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
/// regression test (`testing/lang/test_witness_independence.py`):
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
    let (circuit, public_values) =
        build_circuit(ir, witness, params, CircuitBuilderStage::Keygen, None)?;
    let stats = circuit.builder.statistics();
    Ok(format!(
        "advice={:?} lookup={:?} fixed={} instances={} second_phase={:?}",
        stats.gate.total_advice_per_phase,
        stats.total_lookup_advice_per_phase,
        stats.gate.total_fixed,
        public_values.len(),
        circuit.second_phase.shape(),
    ))
}

/// The circuit of `ir` at `stage` (and its public values): the synthesizer's
/// first-phase cells and second-phase checks, laid out at `params.k`, or
/// pinned to a keygen layout (`pinning`: the circuit parameters and the
/// break points of every phase).
fn build_circuit(
    ir: &IRGraph,
    witness: Option<&ResolvedWitness>,
    params: &ProvingParams,
    stage: CircuitBuilderStage,
    pinning: Option<(CircuitParams, MultiPhaseThreadBreakPoints)>,
) -> Result<(Halo2BaseCircuit, Vec<Fp>), ProvingError> {
    let pinned_table_layout = pinning.as_ref().map(|(p, _)| p.table_layout.clone());
    let mut builder = match (stage, pinning) {
        (CircuitBuilderStage::Prover, Some((circuit_params, break_points))) => {
            BaseCircuitBuilder::prover(circuit_params.base, break_points)
        }
        (_, Some((circuit_params, break_points))) => {
            let mut builder = BaseCircuitBuilder::from_stage(stage);
            builder.set_params(circuit_params.base);
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

    let mut second_phase = circuit::SecondPhase::default();
    let mut tables = circuit::Tables::default();
    let mut ram = circuit::Ram::default();
    let public_values = if let Some(public_values) =
        try_build_public_layout_only(&mut builder, ir, witness)?
    {
        public_values
    } else {
        let mut synth =
            Halo2BaseSynthesizer::new(&mut builder, witness.cloned(), params.clone());
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
        second_phase = std::mem::take(&mut synth.second_phase);
        tables = std::mem::take(&mut synth.tables);
        ram = std::mem::take(&mut synth.ram);
        synth.public_values().to_vec()
    };

    let mut circuit = Halo2BaseCircuit::new(builder, second_phase, tables, ram);
    match pinned_table_layout {
        Some(layout) => circuit.table_layout = layout,
        None if stage != CircuitBuilderStage::Prover => circuit::layout(&mut circuit)?,
        None => {}
    }
    if stage != CircuitBuilderStage::Prover && std::env::var("ZINNIA_HALO2_BASE_DEBUG_CELLS").is_ok() {
        let stats = circuit.builder.statistics();
        eprintln!(
            "[halo2-base cells] advice={:?} lookup={:?} fixed={} instances={} second_phase={:?}",
            stats.gate.total_advice_per_phase,
            stats.total_lookup_advice_per_phase,
            stats.gate.total_fixed,
            public_values.len(),
            circuit.second_phase.shape(),
        );
    }
    Ok((circuit, public_values))
}

fn try_build_public_layout_only(
    builder: &mut BaseCircuitBuilder<Fp>,
    ir: &IRGraph,
    witness: Option<&ResolvedWitness>,
) -> Result<Option<Vec<Fp>>, ProvingError> {
    // Check the complete layout before assigning anything. A failed attempt
    // falls back to full synthesis on this same builder, so even private
    // advice assignments (not just instances) must not escape it.
    let mut available_reads = vec![false; ir.stmts.len()];
    let mut has_public = false;
    for stmt in &ir.stmts {
        match &stmt.ir {
            IR::ReadInteger { is_public, .. } | IR::ReadFloat { is_public, .. } => {
                let Some(read) = available_reads.get_mut(stmt.stmt_id as usize) else {
                    return Ok(None);
                };
                *read = true;
                has_public |= *is_public;
            }
            IR::ExposePublicI | IR::ExposePublicF => {
                if stmt.arguments.len() != 1
                    || !available_reads.get(stmt.arguments[0] as usize).copied().unwrap_or(false)
                {
                    return Ok(None);
                }
                has_public = true;
            }
            IR::StructuralPredicate { .. } | IR::ScalarPrecondition { .. } | IR::ArrayFact(_) => {}
            _ => return Ok(None),
        }
    }
    if !has_public {
        return Ok(None);
    }

    let mut reads: Vec<Option<(AssignedValue<Fp>, Fp)>> = vec![None; ir.stmts.len()];
    let mut public_values = Vec::new();
    let mut pending_instances: Vec<AssignedValue<Fp>> = Vec::new();

    for stmt in &ir.stmts {
        match &stmt.ir {
            IR::ReadInteger { path, is_public } | IR::ReadFloat { path, is_public } => {
                let value = witness
                    .and_then(|w| w.resolve_path(path).ok())
                    .unwrap_or_else(Fp::zero);
                let assigned = builder.main(0).load_witness(value);
                let idx = stmt.stmt_id as usize;
                reads[idx] = Some((assigned, value));
                if *is_public {
                    pending_instances.push(assigned);
                    public_values.push(value);
                }
            }
            IR::ExposePublicI | IR::ExposePublicF => {
                let (assigned, value) = reads[stmt.arguments[0] as usize]
                    .expect("public-layout preflight checked the read");
                pending_instances.push(assigned);
                public_values.push(value);
            }
            | IR::StructuralPredicate { .. }
            | IR::ScalarPrecondition { .. }
            | IR::ArrayFact(_) => {}
            _ => unreachable!("public-layout preflight checked every statement"),
        }
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
