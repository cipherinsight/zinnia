//! Halo2 (IPA) proving backend.
//!
//! Implements `ProverBackend` using the zcash/halo2 proving system with
//! IPA polynomial commitment (no trusted setup required).
//!
//! # Cross-process verification model (D' — re-keygen on verify with cache)
//!
//! zcash/halo2 v0.3 has no public `VerifyingKey` serde, so this backend
//! does **not** serialize the VK directly. Instead, the IR (along with the
//! `precision_bits` / `lookup_bits` params) is shipped in
//! `ProofArtifact.vk_bytes` under the `format = "halo2-ipa-rekeygen-v1"`
//! discriminator, and `verify` re-runs `keygen_vk` to recover the VK from
//! that IR. A process-local `(ir_hash, k, precision_bits, lookup_bits) →
//! (Arc<Params>, Arc<VerifyingKey>)` cache (see `vk_cache.rs`) amortizes
//! the keygen cost across repeated verifies of the same circuit shape —
//! which is the dominant pattern in the differential fuzzer's prover-pool
//! topology and in on-disk artifact replay.
//!
//! See `compiler.halo2-cross-process-vk-serialization` (card README) for
//! the full bump-vs-rekeygen tradeoff analysis. Short version:
//! "PSE 0.4 native serde" turned out to be a git dep (not a crates.io
//! version bump), the published crates.io fork (`halo2-axiom`) is
//! KZG/BN254-only which would drop our IPA/Pasta trusted-setup-free
//! property, and custom serde over zcash 0.3 internals would have
//! required forking the upstream crate. Re-keygen on verify is local,
//! upstream-clean, and amortizes well.
//!
//! ## Soundness invariant (load-bearing for D')
//!
//! D's self-binding between the shipped IR and the proof relies on the
//! standard halo2 pattern: `VerifyingKey::transcript_repr` is absorbed
//! into the Fiat-Shamir transcript at both prove and verify time (via
//! `VerifyingKey::hash_into`, called at the start of `verify_proof`).
//! Verified in `halo2_proofs-0.3.2/src/plonk.rs` (the `transcript_repr`
//! field is computed from the pinned VK at `from_parts` time and absorbed
//! via `common_scalar` in `hash_into`). Equivalent invariant holds in PSE
//! halo2. If a future halo2 refactor ever moved that bind out of the
//! transcript, D' would silently lose self-binding — same risk as the
//! PSE-bump direction would carry, but documented here because the
//! re-keygen approach makes it more visible.
//!
//! ## Setup-fixed-ness statement (`length-hiding` lens)
//!
//! The `VerifyingKey` is determined purely by `(IRGraph shape,
//! ProvingParams { k, precision_bits, lookup_bits })`. No witness- or
//! input-dependent state enters the VK — verified by reading
//! `VerifyingKey::from_parts` in halo2 0.3.2 (`domain`,
//! `fixed_commitments`, `permutation`, `cs` all derive from
//! `Circuit::configure` + `Circuit::synthesize(without_witnesses)`). So
//! the cross-process VK transport (whether via byte serialization or via
//! re-keygen-from-IR as here) does not introduce any new disclosure path
//! beyond what the existing circuit shape already commits to.

pub mod artifact_format;
pub mod circuit;
pub mod config;
pub mod minimal;
pub mod synthesizer;
pub mod vk_cache;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Instant;

use halo2_proofs::halo2curves::bn256::{Fr as Fp, G1Affine as EqAffine};
use halo2_proofs::halo2curves::ff::PrimeField;
use halo2_proofs::{
    dev::MockProver,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::ipa::{
        commitment::{IPACommitmentScheme, ParamsIPA},
        multiopen::{ProverIPA, VerifierIPA},
        strategy::SingleStrategy,
    },
    poly::VerificationStrategy,
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use rand::rngs::OsRng;

use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};

use self::artifact_format::{
    compute_ir_hash, decode as decode_payload, encode as encode_payload, Halo2IpaRekeygenV1Payload,
    HALO2_IPA_REKEYGEN_V1,
};
use self::circuit::{params_for_ir, ZinniaCircuit};
use self::config::ZinniaConfig;
use self::synthesizer::Halo2Synthesizer;
use self::vk_cache::{
    cache_get, cache_insert_after_miss, cache_warm, ipa_params_get_or_create,
    proving_key_cache_get, proving_key_cache_insert,
};

/// The halo2-IPA proving backend.
pub struct Halo2ProverBackend;

// ── Public-input serde ──────────────────────────────────────────────────
//
// Each Fp is serialized via its 32-byte canonical little-endian repr,
// then hex-encoded. This matches `Fp::to_repr` / `Fp::from_repr` round-
// trip semantics.

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

// ── Public-input extraction ─────────────────────────────────────────────
//
// We re-walk the IR once with a standalone `Halo2Synthesizer` (no halo2
// region — phase 1 only) and read out the `public_values` it collected.
// This avoids threading state through halo2's `Circuit::synthesize`,
// which gives us no return channel.

fn collect_public_values(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    params: &ProvingParams,
) -> Result<Vec<Fp>, ProvingError> {
    // Build a throwaway ConstraintSystem just to get a ZinniaConfig.
    // We never use this CS for anything that produces constraints —
    // the synthesizer is run in record-only mode and discarded.
    let mut cs = halo2_proofs::plonk::ConstraintSystem::<Fp>::default();
    let circuit_params = params_for_ir(ir);
    let config =
        ZinniaConfig::configure_with_wide_select_many(&mut cs, circuit_params.wide_select_many);

    let mut synth = Halo2Synthesizer::new(config, Some(witness.clone()), params.clone());
    interpret_ir(ir, &mut synth)
        .map_err(|e| ProvingError::synthesis(format!("Public-input scan failed: {}", e)))?;

    // ── Always-on witness-satisfaction check ────────────────────────────
    //
    // `create_proof` does NOT check that the witness satisfies the gate
    // polynomials — it will happily emit a proof that `verify_proof` later
    // rejects. MockProver caught this at prove time but was removed for
    // perf (commit bb67c32, ~2× prove wall-time). The synthesizer now
    // records, during this same Phase-1 walk, the first assert/bit gate
    // whose concrete witness violates the gate; surface it here so an
    // unsatisfiable witness (e.g. `assert(False)`) fails fast rather than
    // producing a valid-looking artifact for an unsatisfiable circuit.
    if let Some(detail) = synth.first_unsat() {
        return Err(ProvingError::ProvingFailed {
            detail: detail.to_string(),
        });
    }

    Ok(synth.public_values().to_vec())
}

impl ProverBackend for Halo2ProverBackend {
    fn name(&self) -> &'static str {
        "halo2-ipa"
    }

    fn estimate_params(&self, ir: &IRGraph) -> Result<ProvingParams, ProvingError> {
        // Fast path: kernels that touch no gates (only reads, constants,
        // exposes, zkram allocations) — and where the 1-column layout
        // doesn't oversize k — run on the 2-column MinimalCircuit. The
        // eligibility check inside `is_minimal_eligible` already guards
        // against the k-regressing case.
        if crate::prove::halo2::minimal::is_minimal_eligible(ir) {
            let (advice_rows, instance_rows) = crate::prove::halo2::minimal::estimate_rows(ir);
            let needed = advice_rows.max(instance_rows).max(16) + 8;
            let k = ((needed as f64).log2().ceil() as u32).max(4);
            return Ok(ProvingParams {
                k,
                ..ProvingParams::default()
            });
        }
        // Run Phase 1 of synthesis as a dry pass to count rows exactly.
        // Phase 1 (`interpret_ir`) just records gate ops into a Vec and
        // bumps `offset` per row — no FFTs, no commitments. Cost is
        // a few ms even for arithmetic-heavy IR; the savings from a
        // tighter k pay it back many times over in MSM time.
        use halo2_proofs::halo2curves::bn256::Fr as Fp;
        use halo2_proofs::plonk::ConstraintSystem;

        let default_params = ProvingParams::default();
        let mut meta = ConstraintSystem::<Fp>::default();
        let circuit_params = params_for_ir(ir);
        let config =
            crate::prove::halo2::config::ZinniaConfig::configure_with_wide_select_many(
                &mut meta,
                circuit_params.wide_select_many,
            );
        let mut synth = crate::prove::halo2::synthesizer::Halo2Synthesizer::new(
            config,
            None,
            default_params.clone(),
        );
        crate::prove::interpreter::interpret_ir(ir, &mut synth)?;

        // Total rows needed = max(main advice rows, range-table rows,
        // instance-column rows). The instance column needs one usable
        // slot per ExposePublic op, which can dominate for output-heavy
        // kernels (pure reshape, identity copy, etc.).
        let main_rows = synth.row_count();
        let table_rows = if synth.uses_range_lookup() {
            1usize << default_params.lookup_bits.min(16)
        } else {
            1
        };
        let instance_rows = synth.instance_rows();
        // +8 for halo2's blinding/unusable rows at the tail of every column.
        let needed = main_rows.max(table_rows).max(instance_rows) + 8;
        let k = ((needed as f64).log2().ceil() as u32).max(4);
        Ok(ProvingParams {
            k,
            ..default_params
        })
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
        self.prove_inner(ir, witness, params, &ir_stmts, &ir_hash)
    }

    fn prove_with_cached_artifact(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
        ir_stmts: &[serde_json::Value],
        ir_hash: &str,
    ) -> Result<ProofArtifact, ProvingError> {
        // Hot path: the caller-cached `ir_stmts` + `ir_hash` flow through
        // by reference all the way down. No 40 k-Value Vec clone here.
        self.prove_inner(ir, witness, params, ir_stmts, ir_hash)
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        self._verify_impl(artifact)
    }
}

impl Halo2ProverBackend {
    /// Shared prove implementation invoked from both `prove` (which
    /// derives `ir_stmts` + `ir_hash` on the fly) and
    /// `prove_with_cached_artifact` (which receives them from the caller).
    fn prove_inner(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
        ir_stmts: &[serde_json::Value],
        _ir_hash: &str,
    ) -> Result<ProofArtifact, ProvingError> {
        let canonical_ir_hash = compute_ir_hash(ir_stmts)
            .map_err(|e| ProvingError::other(format!("ir_hash compute failed: {}", e)))?;
        let ir_hash = canonical_ir_hash.as_str();
        let k = params.k;
        let ipa_params = ipa_params_get_or_create(k);

        // Fast path: 2-column MinimalCircuit for read/constant/expose-only
        // kernels (and only where 1-column packing doesn't oversize k).
        if crate::prove::halo2::minimal::is_minimal_eligible(ir) {
            return prove_minimal(ir, witness, params, k, &ipa_params, ir_hash, ir_stmts);
        }

        let circuit = ZinniaCircuit {
            ir: ir.clone(),
            witness: Some(witness.clone()),
            params: params.clone(),
        };

        // Collect public inputs by re-walking the IR. This walk also runs
        // the always-on witness-satisfaction check (assert/bit gate cells
        // vs. the concrete witness) and returns `ProvingFailed` on an
        // unsatisfiable witness — the cheap replacement for the removed
        // MockProver pass (see `collect_public_values`). The full
        // MockProver doubled prove wall-time; this check is O(asserts) and
        // piggybacks on a walk we already do.
        let public_inputs: Vec<Fp> = collect_public_values(ir, witness, params)?;

        // Generate or reuse the proving key for this exact proof shape. The
        // key is conservative: canonical IR hash plus every ProvingParams
        // component that shapes Halo2 constraints.
        let pk = if let Some(pk) =
            proving_key_cache_get(ir_hash, k, params.precision_bits, params.lookup_bits)
        {
            pk
        } else {
            let vk =
                keygen_vk(&*ipa_params, &circuit).map_err(|e| ProvingError::ProvingFailed {
                    detail: format!("keygen_vk failed: {:?}", e),
                })?;
            let pk =
                keygen_pk(&*ipa_params, vk, &circuit).map_err(|e| ProvingError::ProvingFailed {
                    detail: format!("keygen_pk failed: {:?}", e),
                })?;
            let pk = Arc::new(pk);
            proving_key_cache_insert(
                ir_hash,
                k,
                params.precision_bits,
                params.lookup_bits,
                Arc::clone(&pk),
            );
            pk
        };

        // Create proof — apples-to-apples timer (matches what
        // halo2-graph's scaffold measures: just create_proof +
        // transcript.finalize, excluding circuit setup, witness JSON
        // parse, artifact assembly, and Python boundary cost).
        let t_create = std::time::Instant::now();
        let mut transcript: Blake2bWrite<Vec<u8>, EqAffine, Challenge255<_>> =
            TranscriptWriterBuffer::<Vec<u8>, EqAffine, _>::init(Vec::new());
        create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
            &ipa_params,
            &pk,
            &[circuit],
            &[&[&public_inputs]],
            OsRng,
            &mut transcript,
        )
        .map_err(|e| ProvingError::ProvingFailed {
            detail: format!("create_proof failed: {:?}", e),
        })?;
        let proof_bytes = transcript.finalize();
        let prove_seconds = t_create.elapsed().as_secs_f64();

        // ── Build the halo2-ipa-rekeygen-v1 payload ────────────────────
        //
        // We ship the IR + the params components that affect circuit
        // shape (precision_bits / lookup_bits) inside `vk_bytes`. `k`
        // lives at the top of `ProofArtifact` already, so we don't
        // duplicate it. The verifier re-keygens from this payload —
        // see the module-level doc-comment on the soundness model.
        let payload = Halo2IpaRekeygenV1Payload {
            ir_stmts: ir_stmts.to_vec(),
            precision_bits: params.precision_bits,
            lookup_bits: params.lookup_bits,
        };
        let vk_bytes = encode_payload(&payload).map_err(|e| {
            ProvingError::other(format!(
                "vk_bytes (halo2-ipa-rekeygen-v1) encode failed: {}",
                e
            ))
        })?;

        // Warm the verify-side cache with the (params, vk) we already
        // computed. This is a pure optimization — the verify path would
        // re-keygen on first lookup anyway — but it avoids the redundant
        // keygen when prove and verify happen in the same process (e.g.
        // the differential fuzzer's child-process model). The cost is
        // one extra Arc clone of objects we'd otherwise drop. Safe under
        // the same Fiat-Shamir self-binding invariant documented at the
        // module level: only artifacts whose ir_hash matches the cache
        // key can use the cached entry, and a wrong-IR artifact is
        // rejected by IrHashMismatch before reaching this slot.
        cache_warm(
            ir_hash,
            k,
            params.precision_bits,
            params.lookup_bits,
            Arc::clone(&ipa_params),
            Arc::new(pk.get_vk().clone()),
        );

        Ok(ProofArtifact {
            backend: "halo2-ipa".to_string(),
            format: HALO2_IPA_REKEYGEN_V1.to_string(),
            vk_bytes,
            ir_hash: ir_hash.to_string(),
            proof_bytes: hex_encode(&proof_bytes),
            public_values: public_inputs.iter().map(encode_fp).collect(),
            k,
            prove_seconds,
        })
    }
}

// The trait `verify` lives on the ProverBackend impl above; the prove
// path (`prove_inner`) lives on the inherent impl below. The two impls
// are kept separate so prove_inner can have full access to the heavy
// halo2 types without needing to fit the trait's narrower API.
impl Halo2ProverBackend {
    fn _verify_impl(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        // Only the v1 format is understood by this build. Empty / missing
        // `format` is rejected with a clear error rather than silently
        // attempting a legacy parse — the previous in-process-handle path
        // was removed in this card.
        if artifact.format != HALO2_IPA_REKEYGEN_V1 {
            return Err(ProvingError::VerificationFailed {
                detail: format!(
                    "Unsupported halo2-ipa artifact format: {:?}. \
                     This build only handles {:?}. \
                     (Legacy in-process handle format was removed by \
                     compiler.halo2-cross-process-vk-serialization.)",
                    artifact.format, HALO2_IPA_REKEYGEN_V1
                ),
            });
        }

        let proof_bytes =
            hex_decode(&artifact.proof_bytes).map_err(|e| ProvingError::VerificationFailed {
                detail: format!("Invalid proof hex: {}", e),
            })?;

        // Reconstruct public inputs from the artifact.
        let public_inputs: Vec<Fp> = artifact
            .public_values
            .iter()
            .map(|s| decode_fp(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ProvingError::VerificationFailed {
                detail: format!("Invalid public-input encoding: {}", e),
            })?;

        // Decode the vk_bytes payload (IR + params shape components).
        let payload =
            decode_payload(&artifact.vk_bytes).map_err(|e| ProvingError::VerificationFailed {
                detail: format!("Failed to decode halo2-ipa-rekeygen-v1 vk_bytes: {}", e),
            })?;

        // ── IR-hash binding check (before proof verification) ──────────
        //
        // Recompute the IR hash from the payload and require it to match
        // the artifact's recorded `ir_hash` *before* spending any keygen /
        // proof-verify work. A mismatch means the verifier was handed an
        // artifact whose advertised circuit (`ir_hash`) differs from the IR
        // actually shipped in `vk_bytes` — i.e. a proof for a DIFFERENT
        // circuit than the IR the caller thinks they are checking. The
        // Fiat-Shamir VK-bind inside `verify_proof` would also reject a
        // wrong-IR proof at the transcript level (see the module soundness
        // note), but we surface the typed `IrHashMismatch` here so the
        // failure is attributable to artifact-transport rather than
        // masquerading as a generic proof-invalid. Returning the typed
        // error (not a `VerifyResult { valid: false }`) lets the
        // differential fuzzer's auto-filer route it as `ir_mismatch`.
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
        let cache_ir_hash = computed_ir_hash.as_str();

        // ── VK lookup (or re-keygen on miss) ───────────────────────────
        let k = artifact.k;
        let (ipa_params, vk) = if let Some(entry) = cache_get(
            cache_ir_hash,
            k,
            payload.precision_bits,
            payload.lookup_bits,
        ) {
            entry
        } else {
            // Cache miss: reconstruct the IR and re-keygen. Dispatch
            // between ZinniaCircuit and MinimalCircuit identically to
            // the prove side — eligibility is deterministic from IR
            // shape so both sides agree without an explicit flag.
            let graph = IRGraph::import_stmts(&payload.ir_stmts).map_err(|e| {
                ProvingError::VerificationFailed {
                    detail: format!("IR import failed: {}", e),
                }
            })?;
            let recovered_params = ProvingParams {
                k,
                precision_bits: payload.precision_bits,
                lookup_bits: payload.lookup_bits,
            };
            let ipa_params = ipa_params_get_or_create(k);

            let keygen_start = Instant::now();
            let vk = if crate::prove::halo2::minimal::is_minimal_eligible(&graph) {
                let circuit = crate::prove::halo2::minimal::MinimalCircuit::shape_only(&graph);
                let _ = recovered_params; // shape-only path doesn't need params
                keygen_vk(&*ipa_params, &circuit).map_err(|e| ProvingError::VerificationFailed {
                    detail: format!("Re-keygen (minimal) failed on verify: {:?}", e),
                })?
            } else {
                let circuit = ZinniaCircuit {
                    ir: graph,
                    witness: None,
                    params: recovered_params,
                };
                keygen_vk(&*ipa_params, &circuit).map_err(|e| ProvingError::VerificationFailed {
                    detail: format!("Re-keygen failed on verify: {:?}", e),
                })?
            };
            let keygen_duration = keygen_start.elapsed();

            let vk_arc = Arc::new(vk);
            cache_insert_after_miss(
                cache_ir_hash,
                k,
                payload.precision_bits,
                payload.lookup_bits,
                Arc::clone(&ipa_params),
                Arc::clone(&vk_arc),
                keygen_duration,
            );
            (ipa_params, vk_arc)
        };

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
            &[&[&public_inputs]],
            &mut transcript,
        ) {
            Ok(_) => Ok(VerifyResult::ok()),
            Err(e) => Ok(VerifyResult::invalid(format!(
                "verify_proof failed: {:?}",
                e
            ))),
        }
    }
}

/// Prove a minimal-eligible circuit (no gates — just reads / constants /
/// exposes). Cuts ~3-4× MSM cost vs the full circuit by configuring only
/// 1 advice + 1 instance column. Verifier-side dispatch in `verify`
/// mirrors this so the two stay in sync.
fn prove_minimal(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    params: &ProvingParams,
    k: u32,
    ipa_params: &Arc<ParamsIPA<EqAffine>>,
    ir_hash: &str,
    ir_stmts: &[serde_json::Value],
) -> Result<ProofArtifact, ProvingError> {
    use crate::prove::halo2::minimal::MinimalCircuit;

    // Pre-resolve the IR + witness into the slim circuit shape (row_values,
    // exposes, public_inputs). Single IR walk per prove — synthesize then
    // just iterates Vec`s. Saves ~250 ms on output-heavy kernels like 0387
    // by keeping halo2's `create_proof` working-set small (no IRGraph
    // HashMaps / in_links Vec`s dragged through the layouter closures).
    let (row_values, exposes, public_inputs) = MinimalCircuit::resolve(ir, witness, params);
    let circuit = MinimalCircuit {
        row_values,
        exposes,
        witnesses_present: true,
    };

    // No witness-satisfaction check here: minimal-eligible kernels are
    // gate-free (reads / constants / exposes only — see
    // `is_minimal_eligible`), so they contain no assert/bit constraint a
    // witness could violate. The always-on check lives on the full
    // ZinniaCircuit path (`collect_public_values`).

    let pk = if let Some(pk) =
        proving_key_cache_get(ir_hash, k, params.precision_bits, params.lookup_bits)
    {
        pk
    } else {
        let vk = keygen_vk(&**ipa_params, &circuit).map_err(|e| ProvingError::ProvingFailed {
            detail: format!("keygen_vk (minimal) failed: {:?}", e),
        })?;
        let pk =
            keygen_pk(&**ipa_params, vk, &circuit).map_err(|e| ProvingError::ProvingFailed {
                detail: format!("keygen_pk (minimal) failed: {:?}", e),
            })?;
        let pk = Arc::new(pk);
        proving_key_cache_insert(
            ir_hash,
            k,
            params.precision_bits,
            params.lookup_bits,
            Arc::clone(&pk),
        );
        pk
    };

    let t_create = std::time::Instant::now();
    let mut transcript: Blake2bWrite<Vec<u8>, EqAffine, Challenge255<_>> =
        TranscriptWriterBuffer::<Vec<u8>, EqAffine, _>::init(Vec::new());
    create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
        ipa_params,
        &pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    )
    .map_err(|e| ProvingError::ProvingFailed {
        detail: format!("create_proof (minimal) failed: {:?}", e),
    })?;
    let proof_bytes = transcript.finalize();
    let prove_seconds = t_create.elapsed().as_secs_f64();

    let payload = Halo2IpaRekeygenV1Payload {
        ir_stmts: ir_stmts.to_vec(),
        precision_bits: params.precision_bits,
        lookup_bits: params.lookup_bits,
    };
    let vk_bytes = encode_payload(&payload).map_err(|e| {
        ProvingError::other(format!(
            "vk_bytes (halo2-ipa-rekeygen-v1) encode failed: {}",
            e
        ))
    })?;

    Ok(ProofArtifact {
        backend: "halo2-ipa".to_string(),
        format: HALO2_IPA_REKEYGEN_V1.to_string(),
        vk_bytes,
        ir_hash: ir_hash.to_string(),
        proof_bytes: hex_encode(&proof_bytes),
        public_values: public_inputs.iter().map(encode_fp).collect(),
        k,
        prove_seconds,
    })
}

/// Run the halo2 MockProver on a circuit for testing/validation.
pub fn mock_prove(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    params: &ProvingParams,
    public_inputs: Vec<Fp>,
) -> Result<(), String> {
    let circuit = ZinniaCircuit {
        ir: ir.clone(),
        witness: Some(witness.clone()),
        params: params.clone(),
    };
    let prover = MockProver::run(params.k, &circuit, vec![public_inputs])
        .map_err(|e| format!("MockProver::run failed: {:?}", e))?;
    prover
        .verify()
        .map_err(|e| format!("MockProver verification failed: {:?}", e))
}

// Simple hex encode/decode (avoid adding another dep)
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
