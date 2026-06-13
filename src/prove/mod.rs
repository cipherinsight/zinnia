//! Proving pipeline for Zinnia ZK circuits.
//!
//! # Architecture
//!
//! ```text
//!                              ProverBackend trait
//!         ┌──────────────┬──────────────┴───────────┬──────────────┐
//!         │              │                          │              │
//!  MockProver-    Halo2Prover-            Halo2BaseProver-    NoirProver-
//!    Backend         Backend                  Backend           Backend
//!         │              │                          │              │
//!    Mock-         Halo2-                  Halo2Base-         NoirEmitter
//! Synthesizer    Synthesizer              Synthesizer    (emits Noir source)
//!         │              │                          │
//!         └───── kernel ─┴── (shared Fp arithmetic, quantization, Remez
//!                            polynomials, Poseidon — value-level semantics)
//! ```
//!
//! The mock and halo2 backends share the same computation kernel
//! (`kernel.rs`), ensuring that the mock backend produces the exact same
//! values that halo2 constrains. The halo2-base backend constrains the
//! same value semantics via the `halo2-base` gadget library, and the noir
//! backend emits Noir source (`noir_fpc`) rather than synthesizing a
//! circuit in-process. All four are selected behind the `ProverBackend`
//! trait; the halo2/halo2-base/noir backends are feature-gated.
//!
//! The `Synthesizer` trait is the per-operation dispatch interface.
//! The `ProverBackend` trait is the top-level prove/verify API.

pub mod error;
pub mod field;
pub mod interpreter;
pub mod ir_hash;
pub mod kernel;
pub mod mock;
pub mod preprocess;
pub mod traits;
pub mod types;

#[cfg(feature = "halo2-backend")]
pub mod halo2;

#[cfg(feature = "halo2-base-backend")]
pub mod halo2_base;

#[cfg(feature = "noir-backend")]
pub mod noir;

// Re-export primary types.
pub use error::ProvingError;
pub use traits::{ProverBackend, Synthesizer};
pub use types::{ProofArtifact, ProvingParams, Value, VerifyResult};

/// Create a `ProverBackend` for the given backend name.
///
/// Supported backends:
/// - `"mock"` — fast concrete evaluation, no proof generation
/// - `"halo2"` / `"halo2-ipa"` — real ZK proof via zcash/halo2 IPA
/// - `"halo2-base"` — real ZK proof via halo2-base's auto-packed circuit builder
pub fn create_prover_backend(backend: &str) -> Result<Box<dyn ProverBackend>, ProvingError> {
    match backend {
        "mock" => Ok(Box::new(mock::MockProverBackend)),
        #[cfg(feature = "halo2-backend")]
        "halo2" | "halo2-ipa" => Ok(Box::new(halo2::Halo2ProverBackend)),
        #[cfg(feature = "halo2-base-backend")]
        "halo2-base" => Ok(Box::new(halo2_base::Halo2BaseProverBackend)),
        #[cfg(feature = "noir-backend")]
        "noir" | "noir-ultrahonk" => Ok(Box::new(noir::NoirProverBackend)),
        _ => Err(ProvingError::other(format!(
            "Unknown proving backend: '{}'. Available: mock, halo2-ipa, halo2-base, noir",
            backend
        ))),
    }
}
