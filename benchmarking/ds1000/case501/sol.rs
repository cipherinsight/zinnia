use std::result;

use clap::Parser;
use ethers_core::types::U256;
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
    pub a: Vec<u64>,
    pub b: Vec<u64>,
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
    // load data
    let a: Vec<AssignedValue<F>> = input.a.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    let b: Vec<AssignedValue<F>> = input.b.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    let results: Vec<AssignedValue<F>> = input.results.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    // apply constraints
    for i in 0..3 {
        for j in 0..3 {
            let item_0 = a[i * 3 * 2 + j * 2 + 0];
            let item_1 = a[i * 3 * 2 + j * 2 + 1];
            let mut selected_item = ctx.load_constant(F::ZERO);
            let b_i_j = b[i * 3 + j];
            let b_i_j_eq_0 = gate.is_equal(ctx, b_i_j, Constant(F::ZERO));
            let b_i_j_eq_1 = gate.is_equal(ctx, b_i_j, Constant(F::ONE));
            let b_i_j_in_range = gate.or(ctx, b_i_j_eq_0, b_i_j_eq_1);
            gate.assert_is_const(ctx, &b_i_j_in_range, &F::ONE);
            let cond_not = gate.not(ctx, b_i_j_eq_0);
            let equal_constraint = gate.is_equal(ctx, item_0, results[i * 3 + j]);
            let constraint = gate.or(ctx, cond_not, equal_constraint);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
            let cond_not = gate.not(ctx, cond_not);
            let equal_constraint = gate.is_equal(ctx, item_1, results[i * 3 + j]);
            let constraint = gate.or(ctx, cond_not, equal_constraint);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
        }
    }
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
