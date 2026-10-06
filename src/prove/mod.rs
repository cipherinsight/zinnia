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
//! circuit in-process. They are selected behind the `ProverBackend`
//! trait; the halo2/halo2-base/noir backends are feature-gated. The legacy
//! halo2 (IPA) backend is retired: `create_prover_backend` refuses it
//! (see [`LEGACY_HALO2_RETIRED`]).
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

/// Why the legacy halo2 (IPA) backend is not selectable.
///
/// Its signed comparisons accept operands that differ by less than 2^56
/// (`halo2::synthesizer::SIGNED_MAGNITUDE_BITS`), so the input-domain checks
/// every circuit with an integer or float input carries (`x >= i64::MIN`,
/// `x <= i64::MAX`, `ir_gen::input_domain`) are unsatisfiable there. The
/// module is kept for its own unit tests and the mock's cross-checks.
pub const LEGACY_HALO2_RETIRED: &str =
    "the legacy halo2 backend cannot prove circuits with 64-bit input checks; use backend='halo2-base'";

/// Create a `ProverBackend` for the given backend name.
///
/// Supported backends:
/// - `"mock"` — fast concrete evaluation, no proof generation
/// - `"halo2-base"` — real ZK proof via halo2-base's auto-packed circuit builder
/// - `"noir"` / `"noir-ultrahonk"` — Noir source, proved with nargo and bb
///
/// `"halo2"` / `"halo2-ipa"` (the legacy backend) is an error naming
/// halo2-base ([`LEGACY_HALO2_RETIRED`]).
pub fn create_prover_backend(backend: &str) -> Result<Box<dyn ProverBackend>, ProvingError> {
    match backend {
        "mock" => Ok(Box::new(mock::MockProverBackend)),
        "halo2" | "halo2-ipa" => Err(ProvingError::other(LEGACY_HALO2_RETIRED.to_string())),
        #[cfg(feature = "halo2-base-backend")]
        "halo2-base" => Ok(Box::new(halo2_base::Halo2BaseProverBackend)),
        #[cfg(feature = "noir-backend")]
        "noir" | "noir-ultrahonk" => Ok(Box::new(noir::NoirProverBackend)),
        _ => Err(ProvingError::other(format!(
            "Unknown proving backend: '{}'. Available: mock, halo2-base, noir",
            backend
        ))),
    }
}

#[cfg(test)]
mod backend_selection_tests {
    use super::*;

    #[test]
    fn legacy_halo2_names_are_rejected_with_the_replacement() {
        for name in ["halo2", "halo2-ipa"] {
            let error = create_prover_backend(name).err().expect("legacy backend must be rejected");
            assert!(error.to_string().contains("backend='halo2-base'"), "{error}");
        }
    }
}
