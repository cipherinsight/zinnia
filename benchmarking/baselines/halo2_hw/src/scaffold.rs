//! Key generation, proving and verification of a handwritten circuit.
//!
//! The proof system is the one Zinnia's halo2-base backend uses
//! (`src/prove/halo2_base/mod.rs`): PSE halo2_proofs v0.3.0 over BN254, IPA
//! commitments (`ParamsIPA::new(k)`, `ProverIPA`/`VerifierIPA`), a Blake2b
//! transcript with `Challenge255`, `OsRng` blinding, and halo2-base's
//! `BaseCircuitBuilder` with keygen-time break points pinned for the prover.
//! The proving window `prove_s` covers `create_proof` plus
//! `transcript.finalize()`, the same two calls Zinnia's `prove_seconds` covers.
//!
//! The three modes communicate through files in a work directory, so each can
//! run in its own process:
//!
//! * `setup`: chooses `k` (unless given), synthesizes the circuit with the
//!   setup input (normally `A`), generates the IPA parameters, the verifying
//!   and proving keys, and writes them with the pinned layout.
//! * `prove`: synthesizes the circuit with the proving input (normally `B`)
//!   against the pinned layout and creates a proof.
//! * `verify`: checks the proof against the instance vector, optionally after
//!   tampering one public value (the run must then fail).

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage};
use halo2_base::gates::flex_gate::MultiPhaseThreadBreakPoints;
use halo2_base::halo2_proofs::{
    arithmetic::Field,
    dev::MockProver,
    halo2curves::bn256::{Fr, G1Affine},
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
use serde_json::{json, Map, Value};

use crate::fixture::Inputs;
use crate::outputs::{Layout, Outputs};
use crate::params::{self, CellStats, UNUSABLE_ROWS};
use crate::{Builder, CircuitFn, Error};

/// How to lay out a synthesis: fresh at a given size, or pinned to keygen.
pub enum Shape {
    Fresh { k: u32, lookup_bits: u32 },
    Pinned { params: BaseCircuitParams, break_points: MultiPhaseThreadBreakPoints },
}

/// Runs the circuit function into a builder for `stage`. For non-prover
/// stages the builder's column layout is computed (`calculate_params`).
pub fn synthesize(
    circuit: CircuitFn,
    inputs: &Inputs,
    stage: CircuitBuilderStage,
    shape: &Shape,
) -> Result<(Builder, Outputs), Error> {
    let mut builder = match shape {
        Shape::Pinned { params, break_points } if stage == CircuitBuilderStage::Prover => {
            BaseCircuitBuilder::prover(params.clone(), break_points.clone())
        }
        Shape::Pinned { params, break_points } => {
            let mut b = BaseCircuitBuilder::from_stage(stage);
            b.set_params(params.clone());
            b.set_break_points(break_points.clone());
            b
        }
        Shape::Fresh { k, lookup_bits } => {
            let mut b = BaseCircuitBuilder::from_stage(stage);
            b.set_k(*k as usize);
            b.set_lookup_bits(*lookup_bits as usize);
            b.set_instance_columns(1);
            b
        }
    };
    let mut outputs = Outputs::default();
    circuit(&mut builder, inputs, &mut outputs)?;
    builder.assigned_instances[0] = outputs.instance_cells().to_vec();
    if stage != CircuitBuilderStage::Prover {
        builder.calculate_params(Some(UNUSABLE_ROWS));
    }
    Ok((builder, outputs))
}

fn instance_values(outputs: &Outputs) -> Vec<Fr> {
    outputs.instance_cells().iter().map(|c| *c.value()).collect()
}

/// Cell counts of the circuit synthesized with `lookup_bits`.
pub fn cell_stats(circuit: CircuitFn, inputs: &Inputs, lookup_bits: u32) -> Result<CellStats, Error> {
    let shape = Shape::Fresh { k: params::K_MAX, lookup_bits };
    let (mut builder, outputs) = synthesize(circuit, inputs, CircuitBuilderStage::Keygen, &shape)?;
    let stats = stats_of(&builder, &outputs);
    builder.clear();
    Ok(stats)
}

fn stats_of(builder: &Builder, outputs: &Outputs) -> CellStats {
    let stats = builder.statistics();
    CellStats {
        advice: stats.gate.total_advice_per_phase.iter().sum(),
        lookup: stats.total_lookup_advice_per_phase.iter().sum(),
        fixed: stats.gate.total_fixed,
        instances: outputs.instance_cells().len(),
    }
}

/// `(k, lookup_bits)`: the given values, or Zinnia's sizing rule.
pub fn choose_size(
    circuit: CircuitFn,
    inputs: &Inputs,
    k: Option<u32>,
    lookup_bits: Option<u32>,
) -> Result<(u32, u32), Error> {
    match (k, lookup_bits) {
        (Some(k), Some(lb)) => Ok((k, lb)),
        (Some(k), None) => Ok((k, params::lookup_bits_for(k))),
        (None, None) => params::choose(|lb| cell_stats(circuit, inputs, lb)),
        (None, Some(_)) => Err(Error::new("--lookup-bits requires --k")),
    }
}

/// Everything `prove` and `verify` need from `setup` besides the keys.
#[derive(Serialize, Deserialize)]
struct Pinning {
    case: String,
    k: u32,
    lookup_bits: u32,
    params: BaseCircuitParams,
    break_points: MultiPhaseThreadBreakPoints,
    layout: Layout,
    cells: CellStats,
}

/// Paths of the artifacts in a work directory.
struct Artifacts {
    dir: PathBuf,
}

impl Artifacts {
    fn new(dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(dir).map_err(|e| Error::new(format!("cannot create {}: {e}", dir.display())))?;
        Ok(Artifacts { dir: dir.to_path_buf() })
    }
    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
    fn create(&self, name: &str) -> Result<BufWriter<File>, Error> {
        let p = self.path(name);
        File::create(&p).map(BufWriter::new).map_err(|e| Error::new(format!("cannot write {}: {e}", p.display())))
    }
    fn open(&self, name: &str) -> Result<BufReader<File>, Error> {
        let p = self.path(name);
        File::open(&p).map(BufReader::new).map_err(|e| Error::new(format!("cannot read {}: {e}", p.display())))
    }
}

const PARAMS_FILE: &str = "params.bin";
const PINNING_FILE: &str = "pinning.json";
const PK_FILE: &str = "pk.bin";
const VK_FILE: &str = "vk.bin";
const PROOF_FILE: &str = "proof.bin";
const INSTANCE_FILE: &str = "instance.json";

/// Wall-clock timings of the named steps, in seconds.
#[derive(Default)]
pub struct Timings(Map<String, Value>);

impl Timings {
    pub fn time<T>(&mut self, name: &str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        self.0.insert(name.to_string(), json!(start.elapsed().as_secs_f64()));
        out
    }
    pub fn into_json(self) -> Value {
        Value::Object(self.0)
    }
}

fn io_err(what: &str) -> impl Fn(std::io::Error) -> Error + '_ {
    move |e| Error::new(format!("{what}: {e}"))
}

/// `setup`: size, synthesize with `inputs`, generate parameters and keys.
pub fn setup(
    case: &str,
    circuit: CircuitFn,
    inputs: &Inputs,
    workdir: &Path,
    k: Option<u32>,
    lookup_bits: Option<u32>,
    t: &mut Timings,
) -> Result<Value, Error> {
    let art = Artifacts::new(workdir)?;
    let (k, lookup_bits) = t.time("choose_k_s", || choose_size(circuit, inputs, k, lookup_bits))?;
    let ipa = t.time("params_s", || ParamsIPA::<G1Affine>::new(k));
    let (mut builder, outputs) = t.time("synthesize_s", || {
        synthesize(circuit, inputs, CircuitBuilderStage::Keygen, &Shape::Fresh { k, lookup_bits })
    })?;
    let vk = t.time("keygen_vk_s", || keygen_vk(&ipa, &builder)).map_err(|e| Error::new(format!("keygen_vk: {e:?}")))?;
    let pk = t.time("keygen_pk_s", || keygen_pk(&ipa, vk, &builder)).map_err(|e| Error::new(format!("keygen_pk: {e:?}")))?;
    let cells = stats_of(&builder, &outputs);
    let pinning = Pinning {
        case: case.to_string(),
        k,
        lookup_bits,
        params: builder.params(),
        break_points: builder.break_points(),
        layout: outputs.layout().clone(),
        cells,
    };
    builder.clear();
    t.time("write_s", || -> Result<(), Error> {
        ipa.write(&mut art.create(PARAMS_FILE)?).map_err(io_err("write params"))?;
        pk.write(&mut art.create(PK_FILE)?, SerdeFormat::RawBytes).map_err(io_err("write pk"))?;
        pk.get_vk().write(&mut art.create(VK_FILE)?, SerdeFormat::RawBytes).map_err(io_err("write vk"))?;
        serde_json::to_writer(art.create(PINNING_FILE)?, &pinning).map_err(|e| Error::new(e.to_string()))?;
        Ok(())
    })?;
    Ok(json!({
        "k": k,
        "lookup_bits": lookup_bits,
        "cells": cells,
        "params": pinning.params,
        "vk_path": art.path(VK_FILE),
        "pk_path": art.path(PK_FILE),
    }))
}

fn load_pinning(art: &Artifacts) -> Result<Pinning, Error> {
    serde_json::from_reader(art.open(PINNING_FILE)?).map_err(|e| Error::new(format!("pinning: {e}")))
}

fn load_params(art: &Artifacts) -> Result<ParamsIPA<G1Affine>, Error> {
    ParamsIPA::<G1Affine>::read(&mut art.open(PARAMS_FILE)?).map_err(io_err("read params"))
}

/// `prove`: synthesize with `inputs` against the pinned layout and prove.
pub fn prove(case: &str, circuit: CircuitFn, inputs: &Inputs, workdir: &Path, t: &mut Timings) -> Result<Value, Error> {
    let art = Artifacts::new(workdir)?;
    let pinning = load_pinning(&art)?;
    if pinning.case != case {
        return Err(Error::new(format!("work directory holds keys for {}, not {case}", pinning.case)));
    }
    let ipa = t.time("load_params_s", || load_params(&art))?;
    let pk = t.time("load_pk_s", || -> Result<ProvingKey<G1Affine>, Error> {
        ProvingKey::read::<_, Builder>(&mut art.open(PK_FILE)?, SerdeFormat::RawBytes, pinning.params.clone())
            .map_err(io_err("read pk"))
    })?;
    let shape = Shape::Pinned { params: pinning.params.clone(), break_points: pinning.break_points.clone() };
    let (builder, outputs) = t.time("witness_s", || synthesize(circuit, inputs, CircuitBuilderStage::Prover, &shape))?;
    let instance = instance_values(&outputs);
    if outputs.layout().segments.len() != pinning.layout.segments.len() || instance.len() != pinning.cells.instances {
        return Err(Error::new("the proving input produced a different output layout than the setup input"));
    }

    let circuits = [builder];
    let proof = t.time("prove_s", || -> Result<Vec<u8>, Error> {
        let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<_>>::init(Vec::new());
        create_proof::<IPACommitmentScheme<G1Affine>, ProverIPA<G1Affine>, _, _, _, _>(
            &ipa,
            &pk,
            &circuits,
            &[&[instance.as_slice()]],
            OsRng,
            &mut transcript,
        )
        .map_err(|e| Error::new(format!("create_proof: {e:?}")))?;
        Ok(transcript.finalize())
    })?;
    let [mut builder] = circuits;
    builder.clear();

    t.time("write_s", || -> Result<(), Error> {
        art.create(PROOF_FILE)?.write_all(&proof).map_err(io_err("write proof"))?;
        let hex: Vec<String> = instance.iter().map(fr_to_hex).collect();
        serde_json::to_writer(art.create(INSTANCE_FILE)?, &hex).map_err(|e| Error::new(e.to_string()))?;
        Ok(())
    })?;
    let publics = pinning.layout.decode(&instance).map_err(Error::new)?;
    Ok(json!({
        "k": pinning.k,
        "lookup_bits": pinning.lookup_bits,
        "cells": pinning.cells,
        "publics": publics,
        "proof_path": art.path(PROOF_FILE),
        "instance_path": art.path(INSTANCE_FILE),
        "proof_bytes": proof.len(),
    }))
}

/// `verify`: check the stored proof against the stored instance vector (or
/// `instance_path`), after adding 1 to position `tamper` if given.
pub fn verify(
    case: &str,
    workdir: &Path,
    instance_path: Option<&Path>,
    tamper: Option<usize>,
    t: &mut Timings,
) -> Result<Value, Error> {
    let art = Artifacts::new(workdir)?;
    let pinning = load_pinning(&art)?;
    if pinning.case != case {
        return Err(Error::new(format!("work directory holds keys for {}, not {case}", pinning.case)));
    }
    let ipa = t.time("load_params_s", || load_params(&art))?;
    let vk = t.time("load_vk_s", || -> Result<VerifyingKey<G1Affine>, Error> {
        VerifyingKey::read::<_, Builder>(&mut art.open(VK_FILE)?, SerdeFormat::RawBytes, pinning.params.clone())
            .map_err(io_err("read vk"))
    })?;
    let proof = std::fs::read(art.path(PROOF_FILE)).map_err(io_err("read proof"))?;
    let instance_file = instance_path.map(Path::to_path_buf).unwrap_or_else(|| art.path(INSTANCE_FILE));
    let hex: Vec<String> = serde_json::from_reader(BufReader::new(
        File::open(&instance_file).map_err(io_err("read instance"))?,
    ))
    .map_err(|e| Error::new(format!("instance: {e}")))?;
    let mut instance: Vec<Fr> = hex.iter().map(|h| fr_from_hex(h)).collect::<Result<_, _>>()?;
    if let Some(i) = tamper {
        let cell = instance
            .get_mut(i)
            .ok_or_else(|| Error::new(format!("tamper index {i} outside the {}-cell instance", hex.len())))?;
        *cell += Fr::ONE;
    }

    let accepted = t.time("verify_s", || {
        let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
        verify_proof::<IPACommitmentScheme<G1Affine>, VerifierIPA<G1Affine>, _, _, SingleStrategy<G1Affine>>(
            &ipa,
            &vk,
            SingleStrategy::new(&ipa),
            &[&[instance.as_slice()]],
            &mut transcript,
        )
        .is_ok()
    });
    let publics = pinning.layout.decode(&instance).unwrap_or_default();
    Ok(json!({
        "k": pinning.k,
        "verified": accepted,
        "tampered_index": tamper,
        "publics": publics,
        "proof_path": art.path(PROOF_FILE),
    }))
}

/// `mock`: synthesize with `inputs` and check every constraint with halo2's
/// `MockProver` (no keys); reports the decoded outputs.
pub fn mock(
    circuit: CircuitFn,
    inputs: &Inputs,
    k: Option<u32>,
    lookup_bits: Option<u32>,
    t: &mut Timings,
) -> Result<Value, Error> {
    let (k, lookup_bits) = t.time("choose_k_s", || choose_size(circuit, inputs, k, lookup_bits))?;
    let (builder, outputs) = t.time("synthesize_s", || {
        synthesize(circuit, inputs, CircuitBuilderStage::Mock, &Shape::Fresh { k, lookup_bits })
    })?;
    let instance = instance_values(&outputs);
    let result = t.time("mock_s", || {
        MockProver::run(k, &builder, vec![instance.clone()]).map(|p| p.verify())
    });
    let failures = match result {
        Err(e) => vec![format!("{e:?}")],
        Ok(Ok(())) => vec![],
        Ok(Err(fs)) => fs.iter().take(5).map(|f| format!("{f}")).collect(),
    };
    let cells = stats_of(&builder, &outputs);
    let mut builder = builder;
    builder.clear();
    Ok(json!({
        "k": k,
        "lookup_bits": lookup_bits,
        "satisfied": failures.is_empty(),
        "failures": failures,
        "cells": cells,
        "publics": outputs.layout().decode(&instance).map_err(Error::new)?,
    }))
}

fn fr_to_hex(x: &Fr) -> String {
    use halo2_base::halo2_proofs::halo2curves::ff::PrimeField;
    x.to_repr().as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn fr_from_hex(s: &str) -> Result<Fr, Error> {
    use halo2_base::halo2_proofs::halo2curves::ff::PrimeField;
    if s.len() != 64 {
        return Err(Error::new(format!("bad field element hex {s:?}")));
    }
    let mut repr = <Fr as PrimeField>::Repr::default();
    for (i, byte) in repr.as_mut().iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| Error::new(e.to_string()))?;
    }
    Option::from(Fr::from_repr(repr)).ok_or_else(|| Error::new(format!("non-canonical field element {s}")))
}
