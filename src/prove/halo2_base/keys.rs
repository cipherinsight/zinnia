//! Key generation, proving and verification as separate steps, and the key
//! files that let each step run in its own process.
//!
//! [`Halo2BaseProverBackend::prove`](super::Halo2BaseProverBackend) regenerates
//! the IPA parameters and runs keygen with the prover's witness on every call,
//! and its `verify` re-runs keygen from the IR shipped in the artifact. That
//! couples one-time setup with every proof and every verification. The
//! functions here split the steps the way a deployment (and the handwritten
//! halo2 baseline, `benchmarking/baselines/halo2_hw`) does: keys once, without
//! a witness; then proofs; then verification against a key the verifier holds.
//!
//! A key directory holds the same files, in the same encodings, as
//! `halo2_hw`'s work directory:
//!
//! * `params.bin`: the IPA parameters (`ParamsIPA::write`);
//! * `pk.bin`: the proving key (`SerdeFormat::RawBytes`; it contains the
//!   verifying key);
//! * `vk.bin`: the verifying key alone (`SerdeFormat::RawBytes`);
//! * `keys.json`: the [`KeyPinning`]: what reading the keys back and proving
//!   with them needs besides the two binary files.
//!
//! A prover reads `params.bin` and `pk.bin`, a verifier `params.bin` and
//! `vk.bin` ([`KeyRole`]); [`Halo2BaseKeys::load`] times both reads.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage};
use halo2_base::gates::flex_gate::MultiPhaseThreadBreakPoints;
use halo2_base::halo2_proofs::{
    halo2curves::bn256::{Fr as Fp, G1Affine as EqAffine},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey},
    poly::{
        commitment::{Params, ParamsProver},
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::{ProverIPA, VerifierIPA},
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    },
    transcript::{Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer},
    SerdeFormat,
};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use super::{build_circuit, without_panic, ESTIMATE_K_MAX};
use crate::circuit_input::ResolvedWitness;
use crate::ir::IRGraph;
use crate::prove::error::ProvingError;
use crate::prove::types::{ProvingParams, VerifyResult};

/// The circuit type keys are read for: reading a key rebuilds the constraint
/// system from the pinned [`BaseCircuitParams`].
type KeyedCircuit = BaseCircuitBuilder<Fp>;

/// The encoding of both keys on disk (the one `halo2_hw` uses): curve points
/// uncompressed in Montgomery form, checked to be on the curve when read.
const KEY_FORMAT: SerdeFormat = SerdeFormat::RawBytes;

const PARAMS_FILE: &str = "params.bin";
const PK_FILE: &str = "pk.bin";
const VK_FILE: &str = "vk.bin";
const PINNING_FILE: &str = "keys.json";

/// What proving and verifying with stored keys needs besides the IPA
/// parameters and the key itself; `keys.json` in a key directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyPinning {
    pub params: ProvingParams,
    /// Column layout of the keygen synthesis. Reading a key back rebuilds the
    /// constraint system from it, and the prover's synthesis is pinned to it.
    pub base_params: BaseCircuitParams,
    /// Thread break points of the keygen synthesis, for the prover.
    pub break_points: MultiPhaseThreadBreakPoints,
    /// `prove::ir_hash::compute_ir_hash` of the circuit the keys were made
    /// for (the hash `CompiledIR` caches).
    pub ir_hash: String,
}

/// The key one party holds.
pub enum CircuitKey {
    /// The prover's key; it contains the verifying key.
    Proving(ProvingKey<EqAffine>),
    /// The verifier's key alone.
    Verifying(VerifyingKey<EqAffine>),
}

/// Which key to read from a key directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyRole {
    Prover,
    Verifier,
}

/// Keys for one compiled circuit and one parameter choice, generated without
/// a witness (exactly the synthesis a verifier would run), or read back from
/// a key directory.
pub struct Halo2BaseKeys {
    pub pinning: KeyPinning,
    pub ipa_params: ParamsIPA<EqAffine>,
    pub key: CircuitKey,
}

/// Seconds spent reading a key directory, per file read.
#[derive(Clone, Copy, Debug)]
pub struct LoadTimings {
    /// `params.bin`.
    pub params_seconds: f64,
    /// `pk.bin` for a prover, `vk.bin` for a verifier.
    pub key_seconds: f64,
}

/// A proof made with [`prove_with_keys`], with its timed phases.
pub struct TimedProof {
    pub proof: Vec<u8>,
    pub public_values: Vec<Fp>,
    /// Witness synthesis: the IR walk filling the pinned circuit builder.
    pub witness_seconds: f64,
    /// Starting the thread pool `create_proof` runs on (outside both windows).
    pub thread_pool_seconds: f64,
    /// `create_proof` plus transcript finalisation.
    pub prove_seconds: f64,
}

impl Halo2BaseKeys {
    pub fn params(&self) -> &ProvingParams {
        &self.pinning.params
    }

    pub fn vk(&self) -> &VerifyingKey<EqAffine> {
        match &self.key {
            CircuitKey::Proving(pk) => pk.get_vk(),
            CircuitKey::Verifying(vk) => vk,
        }
    }

    /// The proving key; an error for keys read for a verifier.
    pub fn pk(&self) -> Result<&ProvingKey<EqAffine>, ProvingError> {
        match &self.key {
            CircuitKey::Proving(pk) => Ok(pk),
            CircuitKey::Verifying(_) => {
                Err(ProvingError::other("these halo2-base keys hold only a verifying key"))
            }
        }
    }

    /// The verifying key in halo2's raw-bytes encoding (for size reports
    /// and out-of-process verifiers).
    pub fn vk_bytes(&self) -> Vec<u8> {
        self.vk().to_bytes(KEY_FORMAT)
    }

    /// Write the keys to `dir` (created if missing). Needs the proving key:
    /// the directory serves both roles.
    pub fn save(&self, dir: &Path) -> Result<(), ProvingError> {
        let pk = self.pk()?;
        std::fs::create_dir_all(dir).map_err(io_error("create", dir))?;
        write_file(&dir.join(PARAMS_FILE), |w| self.ipa_params.write(w))?;
        write_file(&dir.join(PK_FILE), |w| pk.write(w, KEY_FORMAT))?;
        write_file(&dir.join(VK_FILE), |w| pk.get_vk().write(w, KEY_FORMAT))?;
        write_file(&dir.join(PINNING_FILE), |w| Ok(serde_json::to_writer(w, &self.pinning)?))
    }

    /// Read the keys `role` needs from a directory written by [`save`](Self::save).
    pub fn load(dir: &Path, role: KeyRole) -> Result<(Self, LoadTimings), ProvingError> {
        let path = dir.join(PINNING_FILE);
        let pinning: KeyPinning = serde_json::from_reader(open(&path)?)
            .map_err(|e| ProvingError::other(format!("cannot read {}: {e}", path.display())))?;

        let started = Instant::now();
        let path = dir.join(PARAMS_FILE);
        let ipa_params = ParamsIPA::<EqAffine>::read(&mut open(&path)?).map_err(io_error("read", &path))?;
        let params_seconds = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let base_params = pinning.base_params.clone();
        let key = match role {
            KeyRole::Prover => {
                let path = dir.join(PK_FILE);
                CircuitKey::Proving(
                    ProvingKey::read::<_, KeyedCircuit>(&mut open(&path)?, KEY_FORMAT, base_params)
                        .map_err(io_error("read", &path))?,
                )
            }
            KeyRole::Verifier => {
                let path = dir.join(VK_FILE);
                CircuitKey::Verifying(
                    VerifyingKey::read::<_, KeyedCircuit>(&mut open(&path)?, KEY_FORMAT, base_params)
                        .map_err(io_error("read", &path))?,
                )
            }
        };
        let key_seconds = started.elapsed().as_secs_f64();

        if ipa_params.k() != pinning.params.k {
            return Err(ProvingError::other(format!(
                "{} is for k={}, {} says k={}",
                PARAMS_FILE,
                ipa_params.k(),
                PINNING_FILE,
                pinning.params.k
            )));
        }
        let keys = Halo2BaseKeys { pinning, ipa_params, key };
        Ok((keys, LoadTimings { params_seconds, key_seconds }))
    }
}

fn write_file(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> Result<(), ProvingError> {
    let mut writer = File::create(path).map(BufWriter::new).map_err(io_error("create", path))?;
    write(&mut writer).and_then(|()| writer.flush()).map_err(io_error("write", path))
}

fn open(path: &Path) -> Result<BufReader<File>, ProvingError> {
    File::open(path).map(BufReader::new).map_err(io_error("open", path))
}

fn io_error<'a>(action: &'a str, path: &'a Path) -> impl Fn(std::io::Error) -> ProvingError + 'a {
    move |e| ProvingError::other(format!("cannot {action} {}: {e}", path.display()))
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
    let pinning = KeyPinning {
        params: params.clone(),
        base_params: circuit.params(),
        break_points: circuit.break_points(),
        ir_hash,
    };
    circuit.clear();
    Ok(Halo2BaseKeys { pinning, ipa_params, key: CircuitKey::Proving(pk) })
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

/// Prove `ir` on `witness` with keys from [`keygen`] or [`Halo2BaseKeys::load`]
/// (no parameter generation, no keygen).
///
/// The global thread pool `create_proof` runs on is started before the timed
/// window, as `halo2_hw` does: in a fresh process nothing else may have
/// started it yet, and thread creation is not proving.
pub fn prove_with_keys(
    ir: &IRGraph,
    witness: &ResolvedWitness,
    keys: &Halo2BaseKeys,
) -> Result<TimedProof, ProvingError> {
    let pk = keys.pk()?;
    let t_witness = Instant::now();
    let pinning = (keys.pinning.base_params.clone(), keys.pinning.break_points.clone());
    let (circuit, public_values) =
        build_circuit(ir, Some(witness), keys.params(), CircuitBuilderStage::Prover, Some(pinning))?;
    let witness_seconds = t_witness.elapsed().as_secs_f64();

    let t_pool = Instant::now();
    rayon::broadcast(|_| ());
    let thread_pool_seconds = t_pool.elapsed().as_secs_f64();

    let t_prove = Instant::now();
    let instance_refs: Vec<&[Fp]> = vec![public_values.as_slice()];
    let mut transcript: Blake2bWrite<Vec<u8>, EqAffine, Challenge255<_>> =
        TranscriptWriterBuffer::<Vec<u8>, EqAffine, _>::init(Vec::new());
    create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
        &keys.ipa_params,
        pk,
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
    Ok(TimedProof { proof, public_values, witness_seconds, thread_pool_seconds, prove_seconds })
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
