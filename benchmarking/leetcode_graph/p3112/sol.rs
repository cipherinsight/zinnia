use std::result;

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Existing, Witness},
};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;
use snark_verifier_sdk::snark_verifier::halo2_ecc::bigint::negative;
use snark_verifier_sdk::snark_verifier::loader::halo2::IntegerInstructions;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub graph: Vec<u128>,
    pub disappear: Vec<u128>,
    pub answers: Vec<i128>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where  F: BigPrimeField {
    const PRECISION: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let mut poseidon_hasher = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // load graph
    let mut graph: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..100 {
        let value = ctx.load_witness(F::from_u128(input.graph[i]));
        graph.push(value);
    }
    // load disappear
    let mut disappear: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..10 {
        let value = ctx.load_witness(F::from_u128(input.disappear[i]));
        disappear.push(value);
    }
    // load answers
    let mut answers: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..10 {
        if input.answers[i] < 0 {
            let value = ctx.load_witness(F::from_u128((-input.answers[i]) as u128));
            let value = GateInstructions::neg(&gate, ctx, value);
            answers.push(value);

        } else {
            let value = ctx.load_constant(F::from_u128(input.answers[i] as u128));
            answers.push(value);
        }
    }
    // perform the computation
    let negative_one = ctx.load_constant(F::from_u128(1));
    let negative_one = GateInstructions::neg(&gate, ctx, negative_one);
    for k in 0..10 {
        for i in 0..10 {
            for j in 0..10 {
                let idx_graph_i_k = i * 10 + k;
                let idx_graph_k_j = k * 10 + j;
                let idx_graph_i_j = i * 10 + j;
                let graph_i_k = graph[idx_graph_i_k];
                let graph_k_j = graph[idx_graph_k_j];
                let graph_i_k_not_equal_negative_one = gate.is_equal(ctx, graph_i_k, negative_one);
                let graph_i_k_not_equal_negative_one = gate.not(ctx, graph_i_k_not_equal_negative_one);
                let graph_k_j_not_equal_negative_one = gate.is_equal(ctx, graph_k_j, negative_one);
                let graph_k_j_not_equal_negative_one = gate.not(ctx, graph_k_j_not_equal_negative_one);
                let condition = gate.and(ctx, graph_i_k_not_equal_negative_one, graph_k_j_not_equal_negative_one);
                let graph_i_k_plus_graph_k_j = GateInstructions::add(&gate, ctx, graph_i_k, graph_k_j);
                let graph_i_j = graph[idx_graph_i_j];
                let graph_i_j_less_than_graph_i_k_plus_graph_k_j = range_chip.is_less_than(ctx, graph_i_j, graph_i_k_plus_graph_k_j, 128);
                let new_value = gate.select(ctx, graph_i_j, graph_i_k_plus_graph_k_j, graph_i_j_less_than_graph_i_k_plus_graph_k_j);
                graph[idx_graph_i_j] = gate.select(ctx, new_value, graph[idx_graph_i_j], condition);
            }
        }
    }
    // verify answers
    for i in 0..10 {
        let graph_0_i = graph[i];
        let graph_0_i_equals_negative_one = gate.is_equal(ctx, graph_0_i, negative_one);
        let answer_equals_negative_one = gate.is_equal(ctx, answers[i], negative_one);
        let answer_i_equals_graph_0_i = gate.is_equal(ctx, graph_0_i, answers[i]);
        // apply constraint #1
        let disappear_i_equal_graph_0_i: AssignedValue<F> = gate.is_equal(ctx, disappear[i], graph_0_i);
        let disappear_i_gt_graph_0_i: AssignedValue<F> = range_chip.is_less_than(ctx, graph_0_i, disappear[i], 128);
        let disappear_i_gte_graph_0_i = gate.or(ctx, disappear_i_gt_graph_0_i, disappear_i_equal_graph_0_i);
        let cond = gate.and(ctx, disappear_i_gte_graph_0_i, graph_0_i_equals_negative_one);
        let cond_not = gate.not(ctx, cond);
        let constraint = gate.or(ctx, cond_not, answer_i_equals_graph_0_i);
        gate.assert_is_const(ctx, &constraint, &F::ONE);
        // apply constraint #2
        let constraint = gate.or(ctx, cond, answer_equals_negative_one);
        gate.assert_is_const(ctx, &constraint, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
