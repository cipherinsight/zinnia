use std::env::var;
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
    pub matrix: Vec<u128>,
    pub valid: u128,
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

    // load variables
    let matrix = input.matrix.iter().map(|x| ctx.load_witness(F::from_u128(*x))).collect::<Vec<_>>();
    let valid = ctx.load_witness(F::from_u128(input.valid));
    // apply constraints
    let n = 10;
    for i in 0..(n * n) {
        let one_less_than_i = range_chip.is_less_than(ctx, Constant(F::ONE), Constant(F::from(i)), 128);
        let i_less_than_n = range_chip.is_less_than(ctx, Constant(F::from(i)), Constant(F::from(n)), 128);
        let i_eq_n = gate.is_equal(ctx, Constant(F::from(i)), Constant(F::from(n)));
        let cond_rhs = gate.and(ctx, i_less_than_n, i_eq_n);
        let cond_rhs = gate.and(ctx, one_less_than_i, cond_rhs);
        let not_valid = gate.not(ctx, valid);
        let constraint = gate.or(ctx, not_valid, cond_rhs);
        gate.assert_is_const(ctx, &constraint, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
