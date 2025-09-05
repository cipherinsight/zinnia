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

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub data: Vec<u64>,
    pub result: u64,
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

    // load data
    let data = input.data.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    let result = ctx.load_witness(F::from(input.result));

    // verify the solution
    let mut solution = ctx.load_constant(F::ZERO);
    let mut value = ctx.load_constant(F::ZERO);
    for i in 0..(32 * 3) {
        let greater_than = range_chip.is_less_than(ctx, value, data[i], 128);
        value = gate.select(ctx, data[i], value, greater_than);
        solution = gate.select(ctx, Constant(F::from(i as u64)), solution, greater_than);
    }
    let solution_eq_result = gate.is_equal(ctx, solution, result);
    gate.assert_is_const(ctx, &solution_eq_result, &F::ONE);
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
