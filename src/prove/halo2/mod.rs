//! Halo2 (IPA) proving backend.
//!
//! Implements `ProverBackend` using the zcash/halo2 proving system with
//! IPA polynomial commitment (no trusted setup required).

pub mod circuit;
pub mod config;
pub mod synthesizer;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

use halo2_proofs::{
    dev::MockProver,
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, SingleVerifier, VerifyingKey,
    },
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::group::ff::PrimeField;
use pasta_curves::{EqAffine, Fp};
use rand::rngs::OsRng;

use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};
use crate::circuit_input::ResolvedWitness;

use self::circuit::ZinniaCircuit;
use self::config::ZinniaConfig;
use self::synthesizer::Halo2Synthesizer;

/// The halo2-IPA proving backend.
pub struct Halo2ProverBackend;

// ── In-process verifying-key cache ─────────────────────────────────────
//
// zcash/halo2 v0.3 has no built-in `VerifyingKey` serialization. To make
// `prove() → verify()` work end-to-end inside a single process (the
// shape the differential fuzzer needs), we keep the freshly-built VK
// and IPA params in a process-local map keyed by a monotonic id. The
// id is round-tripped through `ProofArtifact::vk_bytes` as a hex-
// encoded big-endian u64.
//
// This is intentionally NOT a substitute for true cross-process VK
// serialization, which is filed as a follow-up. Verifying an artifact
// that did not originate from the same process will return a
// "VK handle not found" error rather than silently succeeding.

type VkEntry = (Arc<Params<EqAffine>>, Arc<VerifyingKey<EqAffine>>);

fn vk_cache() -> &'static Mutex<HashMap<u64, VkEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, VkEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_vk_handle() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn encode_vk_handle(handle: u64) -> String {
    // Big-endian hex so a human reading the artifact sees the value.
    handle
        .to_be_bytes()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

fn decode_vk_handle(s: &str) -> Result<u64, String> {
    let bytes = hex_decode(s)?;
    if bytes.len() != 8 {
        return Err(format!("Expected 8-byte handle, got {} bytes", bytes.len()));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes);
    Ok(u64::from_be_bytes(buf))
}

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
    let config = ZinniaConfig::configure(&mut cs);

    let mut synth = Halo2Synthesizer::new(config, Some(witness.clone()), params.clone());
    interpret_ir(ir, &mut synth)
        .map_err(|e| ProvingError::synthesis(format!("Public-input scan failed: {}", e)))?;

    Ok(synth.public_values().to_vec())
}

impl ProverBackend for Halo2ProverBackend {
    fn name(&self) -> &'static str {
        "halo2-ipa"
    }

    fn estimate_params(&self, ir: &IRGraph) -> Result<ProvingParams, ProvingError> {
        let row_estimate = (ir.stmts.len() * 4).max(16);
        let k = ((row_estimate as f64).log2().ceil() as u32).max(4) + 1;
        Ok(ProvingParams {
            k,
            ..ProvingParams::default()
        })
    }

    fn prove(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> Result<ProofArtifact, ProvingError> {
        let k = params.k;
        let ipa_params = Params::<EqAffine>::new(k);

        let circuit = ZinniaCircuit {
            ir: ir.clone(),
            witness: Some(witness.clone()),
            params: params.clone(),
        };

        // Collect public inputs by re-walking the IR.
        let public_inputs: Vec<Fp> = collect_public_values(ir, witness, params)?;

        // ── Witness validation via MockProver ──────────────────────────
        //
        // `create_proof` does not validate that the witness satisfies the
        // gate polynomials — that is the verifier's job. To catch an
        // unsatisfiable witness (e.g. `assert(false)`, a transcendental
        // bracket failure, an out-of-range div_mod remainder) at prove
        // time, run MockProver first and surface its failures as
        // `ProvingError::ProvingFailed`. This is what makes the
        // differential fuzzer's `satisfied` signal meaningful on the
        // halo2 backend.
        //
        // Cost: roughly doubles prove() wall time. The card explicitly
        // chose correctness over speed here.
        {
            let mock_prover = MockProver::run(k, &circuit, vec![public_inputs.clone()])
                .map_err(|e| ProvingError::ProvingFailed {
                    detail: format!("MockProver::run failed during witness validation: {:?}", e),
                })?;
            if let Err(failures) = mock_prover.verify() {
                return Err(ProvingError::ProvingFailed {
                    detail: format!(
                        "Witness does not satisfy circuit constraints: {}",
                        format_mock_failures(&failures)
                    ),
                });
            }
        }

        // Generate keys
        let vk = keygen_vk(&ipa_params, &circuit).map_err(|e| ProvingError::ProvingFailed {
            detail: format!("keygen_vk failed: {:?}", e),
        })?;
        let pk =
            keygen_pk(&ipa_params, vk.clone(), &circuit).map_err(|e| ProvingError::ProvingFailed {
                detail: format!("keygen_pk failed: {:?}", e),
            })?;

        // Create proof
        let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
        create_proof(
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

        // Stash (params, vk) in the process-local cache so verify() can
        // round-trip without true VK serialization.
        let handle = next_vk_handle();
        let entry: VkEntry = (Arc::new(ipa_params), Arc::new(vk));
        vk_cache()
            .lock()
            .expect("vk cache mutex poisoned")
            .insert(handle, entry);

        Ok(ProofArtifact {
            backend: "halo2-ipa".to_string(),
            vk_bytes: encode_vk_handle(handle),
            proof_bytes: hex_encode(&proof_bytes),
            public_values: public_inputs.iter().map(encode_fp).collect(),
            k,
        })
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        let proof_bytes = hex_decode(&artifact.proof_bytes).map_err(|e| {
            ProvingError::VerificationFailed {
                detail: format!("Invalid proof hex: {}", e),
            }
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

        // Look up the VK and IPA params from the in-process cache.
        // True cross-process VK serialization is filed as a follow-up;
        // zcash/halo2 v0.3 has no built-in VK serde.
        let handle = decode_vk_handle(&artifact.vk_bytes).map_err(|e| {
            ProvingError::VerificationFailed {
                detail: format!("Invalid vk handle: {}", e),
            }
        })?;
        let entry = {
            let guard = vk_cache().lock().expect("vk cache mutex poisoned");
            guard.get(&handle).cloned()
        };
        let (ipa_params, vk) = entry.ok_or_else(|| ProvingError::VerificationFailed {
            detail: format!(
                "VK handle {} not found in process-local cache. \
                 Cross-process verification requires VK serialization, \
                 which is not yet implemented.",
                handle
            ),
        })?;

        let mut transcript =
            Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof_bytes.as_slice());
        let strategy = SingleVerifier::new(&ipa_params);

        match verify_proof(
            &ipa_params,
            &vk,
            strategy,
            &[&[&public_inputs]],
            &mut transcript,
        ) {
            Ok(_) => Ok(VerifyResult::ok()),
            Err(e) => Ok(VerifyResult::invalid(format!("verify_proof failed: {:?}", e))),
        }
    }
}

/// Format a list of `VerifyFailure`s for inclusion in a `ProvingError`.
///
/// We truncate the list to keep error messages tractable on circuits
/// with thousands of failing rows.
fn format_mock_failures(failures: &[halo2_proofs::dev::VerifyFailure]) -> String {
    const MAX_SHOWN: usize = 5;
    let n = failures.len();
    let shown = failures
        .iter()
        .take(MAX_SHOWN)
        .map(|f| format!("{:?}", f))
        .collect::<Vec<_>>()
        .join("; ");
    if n > MAX_SHOWN {
        format!("{} failures (showing first {}): {}", n, MAX_SHOWN, shown)
    } else {
        format!("{} failure(s): {}", n, shown)
    }
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
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("Invalid hex: {}", e))
        })
        .collect()
}
