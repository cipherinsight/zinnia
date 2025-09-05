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
    pub solution: Vec<u64>,
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
    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..6 {
        let mut tmp: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..4 {
            tmp.push(ctx.load_witness(F::from(input.solution[i * 4 + j])));
        }
        result.push(tmp);
    }

    // verify the solution
    for i in 0..6 {
        for j in 0..4 {
            let j_eq_data_i = gate.is_equal(ctx, Constant(F::from(j as u64)), data[i]);
            let j_ne_data_i = gate.not(ctx, j_eq_data_i);
            let result_i_j = result[i][j];
            let result_i_j_is_1 = gate.is_equal(ctx, result_i_j, Constant(F::ONE));
            let result_i_j_is_0 = gate.is_equal(ctx, result_i_j, Constant(F::ZERO));
            // apply assertions #1
            let not_cond = gate.not(ctx, j_eq_data_i);
            let constraint = gate.or(ctx, not_cond, result_i_j_is_1);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
            // apply assertions #2
            let not_cond = gate.not(ctx, j_ne_data_i);
            let constraint = gate.or(ctx, not_cond, result_i_j_is_0);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
        }
    }
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
