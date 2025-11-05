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
    pub area: u128,
    pub expected_l: u128,
    pub expected_w: u128,
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
    let area = ctx.load_witness(F::from_u128(input.area));
    let expected_l = ctx.load_witness(F::from_u128(input.expected_l));
    let expected_w = ctx.load_witness(F::from_u128(input.expected_w));
    let mut w = ctx.load_witness(F::from_u128(input.area));
    let mut break_condition = ctx.load_witness(F::from_u128(0));
    for i in 1..401 {
        let var_i = ctx.load_constant(F::from(i));
        let (_, area_mod_i) = range_chip.div_mod_var(ctx, area, var_i, 128, 128);
        let area_mod_i_equals_0 = gate.is_equal(ctx, area_mod_i, Constant(F::ZERO));
        let not_yet_break_out = gate.not(ctx,break_condition);
        let assign_condition = gate.and(ctx, not_yet_break_out, area_mod_i_equals_0);
        w = gate.select(ctx, var_i, w, assign_condition);
        let i_mul_i = gate.mul(ctx, var_i, var_i);
        let i_mul_i_greater_than_area = range_chip.is_less_than(ctx, i_mul_i, area, 128);
        let i_mul_i_greater_than_area = gate.not(ctx, i_mul_i_greater_than_area);
        break_condition = gate.or(ctx, break_condition, i_mul_i_greater_than_area);
    }
    let (area_div_w, remainder) = range_chip.div_mod_var(ctx, area, w, 128, 128);
    gate.assert_is_const(ctx, &remainder, &F::ZERO);
    let answer_l = area_div_w;
    let l_less_than_w = range_chip.is_less_than(ctx, answer_l, w, 128);
    let final_l = gate.select(ctx, w, answer_l, l_less_than_w);
    let final_w = gate.select(ctx, answer_l, w, l_less_than_w);
    let l_correct = gate.is_equal(ctx, final_l, expected_l);
    let w_correct = gate.is_equal(ctx, final_w, expected_w);
    let constraint = gate.and(ctx, l_correct, w_correct);
    gate.assert_is_const(ctx, &constraint, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
