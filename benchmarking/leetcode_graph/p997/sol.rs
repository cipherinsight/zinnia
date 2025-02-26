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
    pub judge_id: u128,
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
    // load judge_id
    let judge_id = ctx.load_witness(F::from_u128(input.judge_id));
    // apply constraints
    for i in 0..10 {
        let judge_id_minus_one = GateInstructions::sub(&gate, ctx, judge_id, Constant(F::ONE));
        let i_equals_judge_id_minus_one = gate.is_equal(ctx, Constant(F::from_u128(i)), judge_id_minus_one);
        for j in 0..10 {
            let j_equals_judge_id_minus_one = gate.is_equal(ctx, Constant(F::from_u128(j)), judge_id_minus_one);
            let i_not_equal_j = gate.is_equal(ctx, Constant(F::from_u128(i)), Constant(F::from_u128(j)));
            let i_not_equal_j = gate.not(ctx, i_not_equal_j);
            let tmp1 = gate.and(ctx, i_not_equal_j, i_equals_judge_id_minus_one);
            let not_tmp1 = gate.not(ctx, tmp1);
            let trust_0 = gate.is_equal(ctx, graph[(i * 10 + j) as usize], Constant(F::ZERO));
            let constraint = gate.or(ctx, not_tmp1, trust_0);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
            let tmp2 = gate.and(ctx, i_not_equal_j, j_equals_judge_id_minus_one);
            let not_tmp2 = gate.not(ctx, tmp2);
            let trust_1 = gate.is_equal(ctx, graph[(i * 10 + j) as usize], Constant(F::ONE));
            let constraint = gate.or(ctx, not_tmp2, trust_1);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
