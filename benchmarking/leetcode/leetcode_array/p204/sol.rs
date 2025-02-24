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
    pub n: u64,
    pub result: u64
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

    // load n and result as witness
    let n = ctx.load_witness(F::from(input.n));
    let result = ctx.load_witness(F::from(input.result));

    // constrain n between 0 and 1001
    range_chip.check_less_than(ctx, n, Constant(F::from(1001)), 128);
    range_chip.check_less_than(ctx, Constant(F::from(0)), n, 128);

    // constrain result == 1 when n == 0 or n == 1
    let n_eq_0 = gate.is_equal(ctx, n, Constant(F::from(0)));
    let n_eq_1 = gate.is_equal(ctx, n, Constant(F::from(1)));
    let n_eq_0_or_1 = gate.or(ctx, n_eq_0, n_eq_1);
    let result_eq_0 = gate.is_equal(ctx, result, Constant(F::from(0)));
    let result_not_eq_0 = gate.not(ctx, result_eq_0);
    let constraint = gate.or(ctx, result_not_eq_0, n_eq_0_or_1);
    gate.assert_is_const(ctx, &constraint, &F::ONE);

    // calculate number of primes
    let mut is_prime = vec![0; 1001];
    let mut number_of_primes = 0;
    for i in 2..1001 {
        if is_prime[i] == 0 {
            number_of_primes += 1;
            for j in (i..1001).step_by(i) {
                is_prime[j] = 1;
            }
        }
        // constrain that result == number_of_primes when n == i
        let i_eq_n = gate.is_equal(ctx, n, Constant(F::from(i as u64)));
        let not_i_eq_n = gate.not(ctx, i_eq_n);
        let result_correct = gate.is_equal(ctx, result, Constant(F::from(number_of_primes as u64)));
        let constraint = gate.or(ctx, not_i_eq_n, result_correct);
        gate.assert_is_const(ctx, &constraint, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
