//! Mock proving backend.
//!
//! Evaluates IR circuits concretely using the same field arithmetic and
//! approximations as the halo2 backend, but without generating ZK proofs.
//! Used for fast development-time testing and validation.

pub mod synthesizer;

#[cfg(test)]
mod tests;

use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::interpreter::interpret_ir;
use crate::prove::traits::ProverBackend;
use crate::prove::types::{ProofArtifact, ProvingParams, VerifyResult};
use crate::circuit_input::ResolvedWitness;

use self::synthesizer::MockSynthesizer;

/// The mock proving backend — evaluates circuits without generating proofs.
pub struct MockProverBackend;

impl ProverBackend for MockProverBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn estimate_params(&self, _ir: &IRGraph) -> Result<ProvingParams, ProvingError> {
        // Mock doesn't need circuit sizing, but we return default params
        // so the same ProvingParams (precision_bits etc.) are used.
        Ok(ProvingParams::default())
    }

    fn prove(
        &self,
        ir: &IRGraph,
        witness: &ResolvedWitness,
        params: &ProvingParams,
    ) -> Result<ProofArtifact, ProvingError> {
        let mut synth = MockSynthesizer::new(witness.clone(), params.clone());
        interpret_ir(ir, &mut synth)?;

        // Build a mock proof artifact — no actual proof, just the results.
        let satisfied = synth.satisfied;
        let public_values: Vec<String> = synth
            .public_outputs
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();

        Ok(ProofArtifact {
            backend: "mock".to_string(),
            vk_bytes: String::new(),
            proof_bytes: if satisfied {
                "mock_satisfied".to_string()
            } else {
                let failures = synth.assertion_failures.join("; ");
                format!("mock_unsatisfied:{}", failures)
            },
            public_values,
            k: params.k,
        })
    }

    fn verify(&self, artifact: &ProofArtifact) -> Result<VerifyResult, ProvingError> {
        // Mock verification: check if the proof string indicates satisfaction.
        if artifact.proof_bytes == "mock_satisfied" {
            Ok(VerifyResult::ok())
        } else if artifact.proof_bytes.starts_with("mock_unsatisfied") {
            Ok(VerifyResult::invalid(
                artifact.proof_bytes.trim_start_matches("mock_unsatisfied:"),
            ))
        } else {
            Err(ProvingError::VerificationFailed {
                detail: "Not a mock proof artifact".to_string(),
            })
        }
    }
}
