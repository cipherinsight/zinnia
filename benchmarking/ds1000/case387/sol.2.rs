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
    pub results: Vec<u64>,
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
    //    "data": [
    //     1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16
    // ],
    // "results": [
        // 1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16
    // ]


    // load data
    let data = input.data.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    let results = input.results.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    // apply constraints
    for i in 0..2 {
        let valid = gate.is_equal(ctx, data[i * 16 + 0], results[i * 16 + 0]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 1], results[i * 16 + 1]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 2], results[i * 16 + 4]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 3], results[i * 16 + 5]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 4], results[i * 16 + 2]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 5], results[i * 16 + 3]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 6], results[i * 16 + 6]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 7], results[i * 16 + 7]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 8], results[i * 16 + 8]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 9], results[i * 16 + 9]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 10], results[i * 16 + 12]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 11], results[i * 16 + 13]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 12], results[i * 16 + 10]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 13], results[i * 16 + 11]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 14], results[i * 16 + 14]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
        let valid = gate.is_equal(ctx, data[i * 16 + 15], results[i * 16 + 15]);
        gate.assert_is_const(ctx, &valid, &F::ONE);
    }
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
