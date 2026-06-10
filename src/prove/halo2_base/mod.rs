//! halo2-base proving backend.
//!
//! This backend uses Axiom's `halo2-base` virtual-region builder rather than
//! Zinnia's hand-recorded halo2 row layout. The goal is to let Zinnia benefit
//! from the same packing strategy used by the handwritten baselines while
//! keeping the semantic oracle in the shared IR interpreter/kernel.

pub mod artifact_format;
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
        // (advice cells, lookup cells, fixed cells, instance count) per lookup_bits
        let mut dry_cache: HashMap<u32, (usize, usize, usize, usize)> = HashMap::new();
        let mut dry_run = |lb: u32| -> Result<(usize, usize, usize, usize), ProvingError> {
            if let Some(v) = dry_cache.get(&lb) {
                return Ok(*v);
            }
            let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Keygen);
            builder.set_k(ESTIMATE_K_MAX as usize);
            builder.set_lookup_bits(lb as usize);
            builder.set_instance_columns(1);
            let dry_params = ProvingParams {
                k: ESTIMATE_K_MAX,
                lookup_bits: lb,
                ..defaults.clone()
            };
            let instances = {
                let mut synth = Halo2BaseSynthesizer::new(&mut builder, None, dry_params);
                interpret_ir(ir, &mut synth)?;
                synth.public_values().len()
            };
            let stats = builder.statistics();
            let entry = (
                stats.gate.total_advice_per_phase.iter().sum::<usize>(),
                stats.total_lookup_advice_per_phase.iter().sum::<usize>(),
                stats.gate.total_fixed,
                instances,
            );
            dry_cache.insert(lb, entry);
            Ok(entry)
        };

        let mut best: Option<(u128, u32, u32)> = None; // (cost, k, lookup_bits)
        for k in ESTIMATE_K_MIN..=ESTIMATE_K_MAX {
            let lookup_bits = defaults.lookup_bits.min(k.saturating_sub(1)).max(1);
            let (advice, lookup, fixed, instances) = dry_run(lookup_bits)?;
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
        let vk = keygen_vk(&*ipa_params, &keygen_circuit).map_err(|e| {
            ProvingError::ProvingFailed {
                detail: format!("halo2-base keygen_vk failed: {:?}", e),
            }
        })?;
        let pk = keygen_pk(&*ipa_params, vk, &keygen_circuit).map_err(|e| {
            ProvingError::ProvingFailed {
                detail: format!("halo2-base keygen_pk failed: {:?}", e),
            }
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

        let graph = IRGraph::import_stmts(&payload.ir_stmts).map_err(|e| {
            ProvingError::VerificationFailed {
                detail: format!("IR import failed: {}", e),
            }
        })?;
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
            Halo2BaseSynthesizer::new(&mut builder, witness.cloned(), params.clone());
        interpret_ir(ir, &mut synth).map_err(|e| {
            ProvingError::synthesis(format!("halo2-base synthesis failed: {}", e))
        })?;
        synth.public_values().to_vec()
    };

    if stage != CircuitBuilderStage::Prover {
        builder.calculate_params(Some(UNUSABLE_ROWS));
    }
    Ok((builder, public_values))
}

fn try_build_public_layout_only(
    builder: &mut BaseCircuitBuilder<Fp>,
    ir: &IRGraph,
    witness: Option<&ResolvedWitness>,
) -> Result<Option<Vec<Fp>>, ProvingError> {
    let mut reads: Vec<Option<(AssignedValue<Fp>, Fp)>> = vec![None; ir.stmts.len()];
    let mut public_values = Vec::new();

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
                    builder.assigned_instances[0].push(assigned);
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
                builder.assigned_instances[0].push(assigned);
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
    Ok(Some(public_values))
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
