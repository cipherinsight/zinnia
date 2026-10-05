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
    pub image: Vec<u128>,
    pub result: Vec<u128>,
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

    // load image and result as witness
    let mut images: Vec<AssignedValue<F>> = Vec::new();
    let mut results: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..100 {
        let image = ctx.load_witness(F::from_u128(input.image[i]));
        let result = ctx.load_witness(F::from_u128(input.result[i]));
        images.push(image);
        results.push(result);
        let constant_1 = Constant(F::from(1));
        let constant_0 = Constant(F::from(0));
        let eq_0 = gate.is_equal(ctx, image, constant_0);
        let eq_1 = gate.is_equal(ctx, image, constant_1);
        let eq_0_or_1 = gate.or(ctx, eq_0, eq_1);
        gate.assert_is_const(ctx, &eq_0_or_1, &F::ONE);
        let eq_0 = gate.is_equal(ctx, result, constant_0);
        let eq_1 = gate.is_equal(ctx, result, constant_1);
        let eq_0_or_1 = gate.or(ctx, eq_0, eq_1);
        gate.assert_is_const(ctx, &eq_0_or_1, &F::ONE);
    }
    // constrain them
    for i in 0..10 {
        for j in 0..10 {
            let idx = i * 10 + j;
            let idx_other = i * 10 + (10 - 1 - j);
            let constant_1 = Constant(F::from(1));
            let rhs = images[idx_other];
            let one_minus = GateInstructions::sub(&gate, ctx, constant_1, rhs);
            let lhs = results[idx];
            let result_eq = gate.is_equal(ctx, lhs, one_minus);
            gate.assert_is_const(ctx, &result_eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
