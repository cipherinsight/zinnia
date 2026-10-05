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
use snark_verifier_sdk::snark_verifier::loader::halo2::IntegerInstructions;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub result: u64,
    pub nums: Vec<u64>,
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

    // load nums
    let mut nums: Vec<AssignedValue<F>> = Vec::new();
    let result = ctx.load_witness(F::from(input.result));
    for i in 0..20 {
        let num = ctx.load_witness(F::from(input.nums[i]));
        nums.push(num);
    }
    let mut values: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..20 {
        let value = ctx.load_witness(F::from(0));
        values.push(value);
    }
    for i in 0..20 {
        let num = nums[i as usize];
        for j in 0..20 {
            let num_eq_j = gate.is_equal(ctx, num, Constant(F::from(j)));
            let added_value = gate.add(ctx, num, values[j as usize]);
            values[j as usize] = gate.select(ctx, added_value, values[j as usize], num_eq_j);
        }
        // num should be in 0..20
        let num_gt_0 = range_chip.is_less_than(ctx, Constant(F::from(0)), num, 128);
        let num_eq_0 = gate.is_equal(ctx, num, Constant(F::from(0)));
        let num_lt_20 = range_chip.is_less_than(ctx, num, Constant(F::from(20)), 128);
        let constraint = gate.or(ctx, num_gt_0, num_eq_0);
        let constraint = gate.and(ctx, constraint, num_lt_20);
        gate.assert_is_const(ctx, &constraint, &F::ONE);
    }
    let mut take = ctx.load_witness(F::from(0));
    let mut skip = ctx.load_witness(F::from(0));
    for i in 0..20 {
        let take_i = gate.add(ctx, skip, values[i as usize]);
        let skip_less_than_take = range_chip.is_less_than(ctx, skip, take, 128);
        let skip_i = gate.select(ctx, take, skip, skip_less_than_take);
        take = take_i;
        skip = skip_i;
    }
    let skip_less_than_take = range_chip.is_less_than(ctx, skip, take, 128);
    let answer = gate.select(ctx, take, skip, skip_less_than_take);
    let answer_equals_result = gate.is_equal(ctx, result, answer);
    gate.assert_is_const(ctx, &answer_equals_result, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
