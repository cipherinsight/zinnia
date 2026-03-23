//! Proving pipeline for Zinnia ZK circuits.
//!
//! # Architecture
//!
//! ```text
//!                         ProverBackend trait
//!                     ┌───────────┴───────────┐
//!                     │                       │
//!              MockProverBackend       Halo2ProverBackend
//!                     │                       │
//!              MockSynthesizer         Halo2Synthesizer
//!                     │                       │
//!                     └─────── kernel ────────┘
//!                       (shared Fp arithmetic,
//!                        quantization, Remez
//!                        polynomials, Poseidon)
//! ```
//!
//! Both backends share the same computation kernel (`kernel.rs`), ensuring
//! that the mock backend produces the exact same values that the halo2
//! backend constrains. The only difference is that halo2 additionally
//! records gate constraints for proof generation.
//!
//! The `Synthesizer` trait is the per-operation dispatch interface.
//! The `ProverBackend` trait is the top-level prove/verify API.

pub mod error;
pub mod field;
pub mod interpreter;
pub mod kernel;
pub mod mock;
pub mod preprocess;
pub mod traits;
pub mod types;

#[cfg(feature = "halo2-backend")]
pub mod halo2;

// Re-export primary types.
pub use error::ProvingError;
pub use traits::{ProverBackend, Synthesizer};
pub use types::{ProofArtifact, ProvingParams, Value, VerifyResult, WitnessInput};

/// Create a `ProverBackend` for the given backend name.
///
/// Supported backends:
/// - `"mock"` — fast concrete evaluation, no proof generation
/// - `"halo2"` / `"halo2-ipa"` — real ZK proof via zcash/halo2 IPA
pub fn create_prover_backend(backend: &str) -> Result<Box<dyn ProverBackend>, ProvingError> {
    match backend {
        "mock" => Ok(Box::new(mock::MockProverBackend)),
        #[cfg(feature = "halo2-backend")]
        "halo2" | "halo2-ipa" => Ok(Box::new(halo2::Halo2ProverBackend)),
        _ => Err(ProvingError::other(format!(
            "Unknown proving backend: '{}'. Available: mock, halo2-ipa",
            backend
        ))),
    }
}
