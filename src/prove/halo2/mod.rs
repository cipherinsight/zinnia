//! Halo2 (IPA) proving backend.
//!
//! Implements `ProverBackend` using the zcash/halo2 proving system with
//! IPA polynomial commitment (no trusted setup required).

pub mod circuit;
pub mod config;
pub mod synthesizer;

#[cfg(test)]
mod tests;

use halo2_proofs::{
    dev::MockProver,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{vesta, EqAffine, Fp};
use rand::rngs::OsRng;

use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult, WitnessInput};

use self::circuit::ZinniaCircuit;

/// The halo2-IPA proving backend.
pub struct Halo2ProverBackend;

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
        witness: &WitnessInput,
        params: &ProvingParams,
    ) -> Result<ProofArtifact, ProvingError> {
        let k = params.k;
        let ipa_params = Params::<EqAffine>::new(k);

        let circuit = ZinniaCircuit {
            ir: ir.clone(),
            witness: Some(witness.clone()),
            params: params.clone(),
        };

        // For now, no public inputs (will be collected during synthesis in future).
        let public_inputs: Vec<Fp> = Vec::new();

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

        // Note: zcash/halo2 v0.3 VerifyingKey doesn't have built-in
        // serialization. For now, we store an empty placeholder.
        // Full serialization will be implemented when we add a custom
        // serde layer or upgrade to a version with serialization support.
        let vk_buf: Vec<u8> = Vec::new(); // TODO: implement VK serialization

        Ok(ProofArtifact {
            backend: "halo2-ipa".to_string(),
            vk_bytes: hex_encode(&vk_buf),
            proof_bytes: hex_encode(&proof_bytes),
            public_values: public_inputs
                .iter()
                .map(|v| format!("{:?}", v))
                .collect(),
            k,
        })
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        let k = artifact.k;
        let ipa_params = Params::<EqAffine>::new(k);

        let proof_bytes = hex_decode(&artifact.proof_bytes).map_err(|e| {
            ProvingError::VerificationFailed {
                detail: format!("Invalid proof hex: {}", e),
            }
        })?;

        // Parse public inputs
        let public_inputs: Vec<Fp> = Vec::new(); // TODO: parse from artifact.public_values

        // TODO: VK deserialization not yet available in zcash/halo2 v0.3.
        // Full verify will be implemented when VK serialization is added.
        // For now, return an error indicating this is not yet supported.
        Err(ProvingError::other(
            "Standalone verification not yet supported — VK serialization pending. \
             Use mock_prove() for testing.",
        ))
    }
}

/// Run the halo2 MockProver on a circuit for testing/validation.
pub fn mock_prove(
    ir: &IRGraph,
    witness: &WitnessInput,
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
