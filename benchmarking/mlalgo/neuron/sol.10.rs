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
    pub training_data: Vec<f64>,
    pub training_labels: Vec<i64>,
    pub testing_data: Vec<f64>,
    pub testing_labels: Vec<i64>,
    pub initial_weights: Vec<f64>,
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
    let training_data = input.training_data.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let training_labels = input.training_labels.iter().map(|x| (if *x >= 0 {ctx.load_witness(F::from_u128(*x as u128))} else {GateInstructions::neg(&gate, ctx, Witness(F::from_u128((-(*x)) as u128)))})).collect::<Vec<_>>();
    let testing_data = input.testing_data.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let testing_labels = input.testing_labels.iter().map(|x| (if *x >= 0 {ctx.load_witness(F::from_u128(*x as u128))} else {GateInstructions::neg(&gate, ctx, Witness(F::from_u128((-(*x)) as u128)))})).collect::<Vec<_>>();
    let mut weights = input.initial_weights.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();

    // do computations
    let negative_one = GateInstructions::neg(&gate, ctx, Constant(F::ONE));
    for _ in 0..50 {
        for i in 0..100 {
            let mut activation = ctx.load_constant(fixed_point_chip.quantization(0.0));
            let tmp1 = fixed_point_chip.qmul(ctx, weights[0], training_data[i * 2 + 0]);
            let tmp2 = fixed_point_chip.qmul(ctx, weights[1], training_data[i * 2 + 1]);
            let tmp = fixed_point_chip.qadd(ctx, tmp1, tmp2);
            activation = fixed_point_chip.qadd(ctx, activation, tmp);
            let activation_greater_than_zero = range_chip.is_less_than(ctx, activation, Constant(fixed_point_chip.quantization(0.0)), 128);
            let activation_greater_than_zero = gate.not(ctx, activation_greater_than_zero);
            let pred = gate.select(ctx, Constant(F::ONE), negative_one, activation_greater_than_zero);
            let pred_not_equal_label = gate.is_equal(ctx, pred, training_labels[i]);
            let pred_not_equal_label = gate.not(ctx, pred_not_equal_label);
            let label_is_1 = gate.is_equal(ctx, training_labels[i], Constant(F::ONE));
            let label_is_n1 = gate.is_equal(ctx, training_labels[i], negative_one);
            let negative_data1 = fixed_point_chip.qsub(ctx, Constant(fixed_point_chip.quantization(0.0)), training_data[i * 2 + 0]);
            let negative_data2 = fixed_point_chip.qsub(ctx, Constant(fixed_point_chip.quantization(0.0)), training_data[i * 2 + 1]);
            let cond1 = gate.and(ctx, label_is_1, pred_not_equal_label);
            let cond2 = gate.and(ctx, label_is_n1, pred_not_equal_label);
            let updated_weights_1 = fixed_point_chip.qadd(ctx, training_data[i * 2 + 0], weights[0]);
            let updated_weights_2 = fixed_point_chip.qadd(ctx, training_data[i * 2 + 1], weights[1]);
            weights[0] = gate.select(ctx, updated_weights_1, weights[0], cond1);
            weights[1] = gate.select(ctx, updated_weights_2, weights[1], cond1);
            let updated_weights_1 = fixed_point_chip.qadd(ctx, negative_data1, weights[0]);
            let updated_weights_2 = fixed_point_chip.qadd(ctx, negative_data2, weights[1]);
            weights[0] = gate.select(ctx, updated_weights_1, weights[0], cond2);
            weights[1] = gate.select(ctx, updated_weights_2, weights[1], cond2);
        }
    }
    for i in 0..2 {
        let mut activation = ctx.load_constant(fixed_point_chip.quantization(0.0));
        let tmp1 = fixed_point_chip.qmul(ctx, weights[0], testing_data[i * 2 + 0]);
        let tmp2 = fixed_point_chip.qmul(ctx, weights[1], testing_data[i * 2 + 1]);
        let tmp = fixed_point_chip.qadd(ctx, tmp1, tmp2);
        activation = fixed_point_chip.qadd(ctx, activation, tmp);
        let activation_greater_than_zero = range_chip.is_less_than(ctx, activation, Constant(fixed_point_chip.quantization(0.0)), 128);
        let activation_greater_than_zero = gate.not(ctx, activation_greater_than_zero);
        let pred = gate.select(ctx, Constant(F::ONE), negative_one, activation_greater_than_zero);
        let pred_equal_label = gate.is_equal(ctx, pred, testing_labels[i]);
        gate.assert_is_const(ctx, &pred_equal_label, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
