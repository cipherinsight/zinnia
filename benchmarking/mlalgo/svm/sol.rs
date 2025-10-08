use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    AssignedValue,
    QuantumCell::Constant,
};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use halo2_base::poseidon::hasher::PoseidonHasher;
use serde::{Serialize, Deserialize};
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub training_x: Vec<Vec<f64>>, // 10 x 2
    pub training_y: Vec<f64>,      // 10
    pub testing_x: Vec<Vec<f64>>,  // 2 x 2
    pub testing_y: Vec<f64>,       // 2
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
) where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range = builder.range_chip();
    let fp = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Hyperparams / sizes
    let train_m = 10usize;
    let test_m = 2usize;
    let steps = 100usize;

    // Constants
    let lr = Constant(fp.quantization(0.05));
    let inv_m = Constant(fp.quantization(1.0 / (train_m as f64)));
    let one = Constant(fp.quantization(1.0));
    let zero = ctx.load_constant(fp.quantization(0.0));
    let thr = Constant(fp.quantization(0.1)); // margin threshold at test time

    // Load data
    let mut tr_x: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(train_m);
    for i in 0..train_m {
        let xi0 = ctx.load_witness(fp.quantization(input.training_x[i][0]));
        let xi1 = ctx.load_witness(fp.quantization(input.training_x[i][1]));
        tr_x.push(vec![xi0, xi1]);
    }
    let tr_y: Vec<AssignedValue<F>> = (0..train_m)
        .map(|i| ctx.load_witness(fp.quantization(input.training_y[i])))
        .collect();

    let mut te_x: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(test_m);
    for i in 0..test_m {
        let xi0 = ctx.load_witness(fp.quantization(input.testing_x[i][0]));
        let xi1 = ctx.load_witness(fp.quantization(input.testing_x[i][1]));
        te_x.push(vec![xi0, xi1]);
    }
    let te_y: Vec<AssignedValue<F>> = (0..test_m)
        .map(|i| ctx.load_witness(fp.quantization(input.testing_y[i])))
        .collect();

    // Parameters: w ∈ R^2, b ∈ R
    let mut w0 = zero;
    let mut w1 = zero;
    let mut b = zero;

    // Training loop (subgradient on hinge + L2)
    for _ in 0..steps {
        let mut gw0 = zero;
        let mut gw1 = zero;
        let mut gb  = zero;

        for i in 0..train_m {
            // score = x·w + b
            let xw0 = fp.qmul(ctx, tr_x[i][0], w0);
            let xw1 = fp.qmul(ctx, tr_x[i][1], w1);
            let s01 = fp.qadd(ctx, xw0, xw1);
            let score = fp.qadd(ctx, s01, b);

            // margin = y * score
            let margin = fp.qmul(ctx, tr_y[i], score);

            // if margin < 1: accumulate -y*x and -y into grads
            let is_hinge = range.is_less_than(ctx, margin, one, 128);
            // Δgw0 = -y * x0 ; Δgw1 = -y * x1 ; Δgb = -y
            let yx0 = fp.qmul(ctx, tr_y[i], tr_x[i][0]);
            let yx1 = fp.qmul(ctx, tr_y[i], tr_x[i][1]);
            let neg_yx0 = fp.neg(ctx, yx0);
            let neg_yx1 = fp.neg(ctx, yx1);
            let neg_y   = fp.neg(ctx, tr_y[i]);

            // conditional add via select: cond? (g+Δ) : g
            let cand_gw0 = fp.qadd(ctx, gw0, neg_yx0);
            gw0 = gate.select(ctx, cand_gw0, gw0, is_hinge);

            let cand_gw1 = fp.qadd(ctx, gw1, neg_yx1);
            gw1 = gate.select(ctx, cand_gw1, gw1, is_hinge);

            let cand_gb = fp.qadd(ctx, gb, neg_y);
            gb = gate.select(ctx, cand_gb, gb, is_hinge);
        }

        // Average hinge part and add L2 term (w) to gw
        let tmp1 = fp.qmul(ctx, gw0, inv_m);
        let tmp2 = fp.qmul(ctx, gw1, inv_m);
        gw0 = fp.qadd(ctx, tmp1, w0);
        gw1 = fp.qadd(ctx, tmp2, w1);
        gb  = fp.qmul(ctx, gb, inv_m);

        // Parameter updates: w -= lr * gw ; b -= lr * gb
        let lr_gw0 = fp.qmul(ctx, lr, gw0);
        let lr_gw1 = fp.qmul(ctx, lr, gw1);
        let lr_gb  = fp.qmul(ctx, lr, gb);

        w0 = fp.qsub(ctx, w0, lr_gw0);
        w1 = fp.qsub(ctx, w1, lr_gw1);
        b  = fp.qsub(ctx, b,  lr_gb);
    }

    // Test-time checks: assert y * (x·w + b) > 0.1
    for i in 0..test_m {
        let xw0 = fp.qmul(ctx, te_x[i][0], w0);
        let xw1 = fp.qmul(ctx, te_x[i][1], w1);
        let s01 = fp.qadd(ctx, xw0, xw1);
        let pred = fp.qadd(ctx, s01, b);
        let margin = fp.qmul(ctx, te_y[i], pred);

        // margin > 0.1  <=>  0.1 < margin
        let ok = range.is_less_than(ctx, thr, margin, 128);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
