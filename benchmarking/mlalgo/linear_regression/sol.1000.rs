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
    pub training_x: Vec<f64>,
    pub training_y: Vec<f64>,
    pub testing_x: Vec<f64>,
    pub testing_y: Vec<f64>,
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
    let training_x = input.training_x.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let testing_x = input.testing_x.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let training_y = input.training_y.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let testing_y = input.testing_y.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();

    // do computations
    let mut bias = ctx.load_constant(fixed_point_chip.quantization(0.0));
    let mut weights = vec![ctx.load_constant(fixed_point_chip.quantization(0.0)); 2];
    let m = 10000;
    for _ in 0..100 {
        let mut errors: Vec<AssignedValue<F>> = Vec::new();
        for i in 0..m {
            let mut pred = ctx.load_constant(fixed_point_chip.quantization(0.0));
            let tmp = fixed_point_chip.qmul(ctx, weights[0], training_x[i * 2 + 0]);
            pred = fixed_point_chip.qadd(ctx, pred, tmp);
            let tmp = fixed_point_chip.qmul(ctx, weights[1], training_x[i * 2 + 1]);
            pred = fixed_point_chip.qadd(ctx, pred, tmp);
            pred = fixed_point_chip.qadd(ctx, pred, bias);
            let loss = fixed_point_chip.qsub(ctx, pred, training_y[i]);
            errors.push(loss);
        }
        let mut dw = vec![ctx.load_constant(fixed_point_chip.quantization(0.0)); 2];
        let mut db = ctx.load_constant(fixed_point_chip.quantization(0.0));
        let one_div_m = fixed_point_chip.qdiv(ctx, Constant(fixed_point_chip.quantization(1.0)), Constant(fixed_point_chip.quantization(m as f64)));
        let mut sum_of_errors = ctx.load_constant(fixed_point_chip.quantization(0.0));
        for i in 0..m {
            sum_of_errors = fixed_point_chip.qadd(ctx, sum_of_errors, errors[i]);
        }
        for i in 0..m {
            let tmp = fixed_point_chip.qmul(ctx, errors[i], training_x[i * 2 + 0]);
            let tmp = fixed_point_chip.qmul(ctx, tmp, one_div_m);
            dw[0] = fixed_point_chip.qadd(ctx, dw[0], tmp);
            let tmp = fixed_point_chip.qmul(ctx, errors[i], training_x[i * 2 + 1]);
            let tmp = fixed_point_chip.qmul(ctx, tmp, one_div_m);
            dw[1] = fixed_point_chip.qadd(ctx, dw[1], tmp);
            let tmp = fixed_point_chip.qmul(ctx, one_div_m, sum_of_errors);
            db = fixed_point_chip.qadd(ctx, db, tmp);
        }
        let tmp1 = fixed_point_chip.qmul(ctx, dw[0], Constant(fixed_point_chip.quantization(0.02)));
        let tmp2 = fixed_point_chip.qmul(ctx, dw[1], Constant(fixed_point_chip.quantization(0.02)));
        weights[0] = fixed_point_chip.qsub(ctx, weights[0], tmp1);
        weights[1] = fixed_point_chip.qsub(ctx, weights[1], tmp2);
        let tmp3 = fixed_point_chip.qmul(ctx, db, Constant(fixed_point_chip.quantization(0.02)));
        bias = fixed_point_chip.qsub(ctx, bias, tmp3);
    }
    println!("weights: {:?}, {:?}, bias: {:?}", fixed_point_chip.dequantization(*weights[0].value()), fixed_point_chip.dequantization(*weights[1].value()), fixed_point_chip.dequantization(*bias.value()));
    // evaluate the model
    let mut errors: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..2 {
        let mut pred = ctx.load_constant(fixed_point_chip.quantization(0.0));
        let tmp = fixed_point_chip.qmul(ctx, weights[0], testing_x[i * 2 + 0]);
        pred = fixed_point_chip.qadd(ctx, pred, tmp);
        let tmp = fixed_point_chip.qmul(ctx, weights[1], testing_x[i * 2 + 1]);
        pred = fixed_point_chip.qadd(ctx, pred, tmp);
        pred = fixed_point_chip.qadd(ctx, pred, bias);
        let loss = fixed_point_chip.qsub(ctx, pred, testing_y[i]);
        errors.push(loss);
    }
    let mut sum_of_errors = ctx.load_constant(fixed_point_chip.quantization(0.0));
    for i in 0..2 {
        let tmp = fixed_point_chip.qmul(ctx, errors[i], errors[i]);
        sum_of_errors = fixed_point_chip.qadd(ctx, sum_of_errors, tmp);
    }
    let total_error = fixed_point_chip.qdiv(ctx, sum_of_errors, Constant(fixed_point_chip.quantization(2.0)));
    println!("total loss: {:?}", fixed_point_chip.dequantization(*total_error.value()));
    let constraint = range_chip.is_less_than(ctx, total_error, Constant(fixed_point_chip.quantization(1.0)), 128);
    gate.assert_is_const(ctx, &constraint, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
