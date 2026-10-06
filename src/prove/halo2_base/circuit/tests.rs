//! Tests of the two-phase circuit with a small check: `y` is a permutation of
//! `x`, by the grand product `prod (gamma - x_i) == prod (gamma - y_i)` at a
//! challenge `gamma` (two equal-length multisets of field elements are equal
//! exactly when these two degree-`n` polynomials in `gamma` are, and two
//! different ones agree on at most `n` points: Schwartz-Zippel).
//!
//! MockProver: the honest witness is accepted; a `y` that is not a
//! permutation, and a wrong value in the challenge cell, are rejected. Real
//! IPA proofs through the backend's stages (keygen without a witness, a
//! prover pinned to the keygen layout, verification): the honest proof
//! verifies, a proof for a non-permutation does not.

use std::sync::Arc;

use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
use halo2_base::gates::flex_gate::{GateChip, GateInstructions, MultiPhaseThreadBreakPoints};
use halo2_base::gates::range::RangeInstructions;
use halo2_base::halo2_proofs::{
    dev::MockProver,
    halo2curves::bn256::{Fr as Fp, G1Affine},
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
    transcript::{Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use halo2_base::{AssignedValue, Context, QuantumCell};
use rand::rngs::OsRng;

use super::{layout, CircuitParams, Halo2BaseCircuit, SecondPhase, SecondPhaseCheck};

const K: usize = 9;
const LOOKUP_BITS: usize = 8;

/// `ys` is a permutation of `xs` (same length), by a grand product at one
/// challenge.
struct Permutation {
    xs: Vec<AssignedValue<Fp>>,
    ys: Vec<AssignedValue<Fp>>,
}

impl SecondPhaseCheck for Permutation {
    fn challenges(&self) -> usize {
        1
    }

    fn build(&self, ctx: &mut Context<Fp>, gate: &GateChip<Fp>, challenges: &[AssignedValue<Fp>]) {
        let gamma = challenges[0];
        let product = |ctx: &mut Context<Fp>, values: &[AssignedValue<Fp>]| {
            let mut acc = ctx.load_constant(Fp::one());
            for v in values {
                let factor = gate.sub(ctx, gamma, *v);
                acc = gate.mul(ctx, acc, factor);
            }
            acc
        };
        let px = product(ctx, &self.xs);
        let py = product(ctx, &self.ys);
        ctx.constrain_equal(&px, &py);
    }
}

/// The circuit over private `xs`, `ys` (first phase: the cells, a range
/// check of `xs[0]`, `xs[0]` public), with the permutation check; laid out
/// by [`layout`], or pinned to a keygen layout in the prover stage.
fn circuit(
    xs: &[u64],
    ys: &[u64],
    stage: CircuitBuilderStage,
    pinning: Option<(CircuitParams, MultiPhaseThreadBreakPoints)>,
) -> (Halo2BaseCircuit, Vec<Fp>) {
    let mut builder = match pinning.clone() {
        Some((params, break_points)) if stage == CircuitBuilderStage::Prover => {
            BaseCircuitBuilder::prover(params.base, break_points)
        }
        _ => {
            let mut b = BaseCircuitBuilder::from_stage(stage).use_k(K).use_lookup_bits(LOOKUP_BITS);
            b.set_instance_columns(1);
            b
        }
    };
    let range = builder.range_chip();
    let ctx = builder.main(0);
    let x = ctx.assign_witnesses(xs.iter().map(|v| Fp::from(*v)));
    let y = ctx.assign_witnesses(ys.iter().map(|v| Fp::from(*v)));
    range.range_check(ctx, x[0], 16);
    let sum = range.gate().sum(ctx, x.iter().map(|c| QuantumCell::Existing(*c)));
    builder.assigned_instances[0].push(sum);
    let publics = vec![*sum.value()];
    let mut second_phase = SecondPhase::default();
    second_phase.push(Arc::new(Permutation { xs: x, ys: y }));
    let mut circuit = Halo2BaseCircuit::new(builder, second_phase);
    match pinning {
        Some((params, _)) if stage != CircuitBuilderStage::Prover => circuit.builder.set_params(params.base),
        Some(_) => {}
        None => layout(&mut circuit),
    }
    (circuit, publics)
}

fn mock_accepts(circuit: &Halo2BaseCircuit, publics: &[Fp]) -> bool {
    MockProver::run(K as u32, circuit, vec![publics.to_vec()]).expect("MockProver runs").verify().is_ok()
}

const XS: [u64; 6] = [3, 1, 4, 1, 5, 9];

#[test]
fn a_permutation_is_accepted_and_a_non_permutation_rejected() {
    let (honest, publics) = circuit(&XS, &[9, 1, 5, 4, 3, 1], CircuitBuilderStage::Mock, None);
    assert_eq!(honest.params().challenges, 1);
    assert_eq!(honest.builder.config_params.num_advice_per_phase.len(), 2, "a second-phase column");
    assert!(mock_accepts(&honest, &publics));
    for ys in [[9, 1, 5, 4, 3, 2], [9, 1, 5, 4, 3, 3], [1, 1, 1, 4, 5, 9]] {
        let (bad, publics) = circuit(&XS, &ys, CircuitBuilderStage::Mock, None);
        assert!(!mock_accepts(&bad, &publics), "ys = {ys:?} accepted");
    }
}

#[test]
fn a_wrong_challenge_cell_is_rejected() {
    let (mut cheating, publics) = circuit(&XS, &[9, 1, 5, 4, 3, 1], CircuitBuilderStage::Mock, None);
    for offset in [Fp::one(), -Fp::one(), Fp::from(1u64 << 40)] {
        cheating.challenge_offset = offset;
        assert!(!mock_accepts(&cheating, &publics), "challenge moved by {offset:?} accepted");
    }
}

#[test]
fn a_circuit_without_checks_is_one_phase() {
    let mut builder = BaseCircuitBuilder::<Fp>::from_stage(CircuitBuilderStage::Mock).use_k(K).use_lookup_bits(LOOKUP_BITS);
    builder.set_instance_columns(1);
    let range = builder.range_chip();
    let ctx = builder.main(0);
    let x = ctx.load_witness(Fp::from(7u64));
    range.range_check(ctx, x, 16);
    let mut circuit = Halo2BaseCircuit::new(builder, SecondPhase::default());
    layout(&mut circuit);
    assert_eq!(circuit.params().challenges, 0);
    assert_eq!(circuit.builder.config_params.num_advice_per_phase.len(), 1);
    assert!(mock_accepts(&circuit, &[]));
}

/// Keys without a witness, a proof pinned to the keygen layout, and
/// verification, as the backend runs them.
fn prove_and_verify(ys: &[u64]) -> bool {
    let ipa = ParamsIPA::<G1Affine>::new(K as u32);
    let (keygen, _) = circuit(&XS, &[9, 1, 5, 4, 3, 1], CircuitBuilderStage::Keygen, None);
    let vk = keygen_vk(&ipa, &keygen).expect("keygen_vk");
    let pk = keygen_pk(&ipa, vk, &keygen).expect("keygen_pk");
    let pinning = (keygen.params(), keygen.break_points());
    let (prover, publics) = circuit(&XS, ys, CircuitBuilderStage::Prover, Some(pinning.clone()));
    let mut transcript = Blake2bWrite::<Vec<u8>, G1Affine, Challenge255<_>>::init(Vec::new());
    create_proof::<IPACommitmentScheme<G1Affine>, ProverIPA<G1Affine>, _, _, _, _>(
        &ipa,
        &pk,
        &[prover],
        &[&[publics.as_slice()]],
        OsRng,
        &mut transcript,
    )
    .expect("create_proof");
    let proof = transcript.finalize();
    // The verifier re-keys from the pinned parameters, as `Halo2BaseKeys::load` does.
    let (verifier_keygen, _) = circuit(&XS, &[9, 1, 5, 4, 3, 1], CircuitBuilderStage::Keygen, Some(pinning));
    let vk = keygen_vk(&ipa, &verifier_keygen).expect("verifier keygen_vk");
    let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
    verify_proof::<IPACommitmentScheme<G1Affine>, VerifierIPA<G1Affine>, _, _, _>(
        &ipa,
        &vk,
        SingleStrategy::new(&ipa),
        &[&[publics.as_slice()]],
        &mut transcript,
    )
    .is_ok()
}

#[test]
fn real_proofs_verify_for_a_permutation_only() {
    assert!(prove_and_verify(&[1, 9, 4, 1, 3, 5]), "honest proof rejected");
    assert!(!prove_and_verify(&[1, 9, 4, 1, 3, 6]), "proof for a non-permutation verified");
}
