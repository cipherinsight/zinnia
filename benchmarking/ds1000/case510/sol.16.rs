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
    pub inputs: Vec<u64>,
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
    let inputs: Vec<AssignedValue<F>> = input.inputs.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    let results: Vec<AssignedValue<F>> = input.results.iter().map(|x| ctx.load_witness(F::from(*x))).collect::<Vec<_>>();
    // apply constraints
    let mut zero_rows = vec![ctx.load_constant(F::ZERO); 5];
    let mut zero_cols = vec![ctx.load_constant(F::ZERO); 96];
    for i in 0..5 {
        let mut tmp = ctx.load_constant(F::ONE);
        for j in 0..96 {
            let is_0 = gate.is_zero(ctx, inputs[i * 96 + j]);
            tmp = gate.and(ctx, tmp, is_0);
        }
        zero_rows[i] = tmp;
    }
    for i in 0..96 {
        let mut tmp = ctx.load_constant(F::ONE);
        for j in 0..5 {
            let is_0 = gate.is_zero(ctx, inputs[j * 96 + i]);
            tmp = gate.and(ctx, tmp, is_0);
        }
        zero_cols[i] = tmp;
    }
    let mut idx = ctx.load_constant(F::ZERO);
    for i in 0..5 {
        for j in 0..96 {
            let cont = gate.or(ctx, zero_rows[i], zero_cols[j]);
            let cont_not = gate.not(ctx, cont);
            let mut target_ans = ctx.load_constant(F::ZERO);
            let idx_lt_0 = range_chip.is_less_than(ctx, idx, Constant(F::ZERO), 128);
            let idx_gte_0 = gate.not(ctx, idx_lt_0);
            let idx_lt_ = range_chip.is_less_than(ctx, idx, Constant(F::from(3 * 94)), 128);
            let constraint = gate.and(ctx, idx_gte_0, idx_lt_);
            let constraint = gate.or(ctx, constraint, cont);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
            for k in 0..(3 * 94) {
                let idx_eq_k = gate.is_equal(ctx, idx, Constant(F::from(k)));
                target_ans = gate.select(ctx, results[k as usize], target_ans, idx_eq_k);
            }
            let equals = gate.is_equal(ctx, target_ans, inputs[i * 96 + j]);
            let constraint = gate.or(ctx, cont, equals);
            gate.assert_is_const(ctx, &constraint, &F::ONE);
            let idx_add_1 = gate.add(ctx, idx.clone(), Constant(F::ONE));
            idx = gate.select(ctx, idx_add_1, idx, cont_not);
        }
    }
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
