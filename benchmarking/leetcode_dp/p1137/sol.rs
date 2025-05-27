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
    pub result: u128,
    pub n: u128,
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

    // load n and result
    let result = ctx.load_witness(F::from_u128(input.result));
    let n = ctx.load_witness(F::from_u128(input.n));

    // constrain result == 0 if n == 0
    let n_is_zero = gate.is_equal(ctx, n, Constant(F::ZERO));
    let n_is_not_zero = gate.not(ctx, n_is_zero);
    let result_is_zero = gate.is_equal(ctx, n, Constant(F::ZERO));
    let constraint = gate.or(ctx, n_is_not_zero, result_is_zero);
    gate.assert_is_const(ctx, &constraint, &F::ONE);

    // constrain result == 1 if n == 1
    let n_is_one = gate.is_equal(ctx, n, Constant(F::ONE));
    let n_is_not_one = gate.not(ctx, n_is_one);
    let result_is_one = gate.is_equal(ctx, result, Constant(F::ONE));
    let constraint = gate.or(ctx, n_is_not_one, result_is_one);
    gate.assert_is_const(ctx, &constraint, &F::ONE);

    // constrain result == 1 if n == 2
    let n_is_two = gate.is_equal(ctx, n, Constant(F::from_u128(2)));
    let n_is_not_two = gate.not(ctx, n_is_two);
    let result_is_one = gate.is_equal(ctx, result, Constant(F::ONE));
    let constraint = gate.or(ctx, n_is_not_two, result_is_one);
    gate.assert_is_const(ctx, &constraint, &F::ONE);

    // constrain others
    let mut a: AssignedValue<F> = ctx.load_constant(F::ZERO);
    let mut b: AssignedValue<F> = ctx.load_constant(F::ONE);
    let mut c: AssignedValue<F> = ctx.load_constant(F::ONE);
    for i in 3..101 {
        let tmp = gate.add(ctx, a, b);
        let tmp = gate.add(ctx, tmp, c);
        a = b;
        b = c;
        c = tmp;
        // constrain result == c if n == i
        let n_is_i = gate.is_equal(ctx, n, Constant(F::from_u128(i)));
        let n_is_not_i = gate.not(ctx, n_is_i);
        let result_is_c = gate.is_equal(ctx, result, c);
        let constraint = gate.or(ctx, n_is_not_i, result_is_c);
        gate.assert_is_const(ctx, &constraint, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
