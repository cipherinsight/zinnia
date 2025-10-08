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
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range = builder.range_chip();
    let fp = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Constants
    let train_m = 10usize;
    let test_m = 2usize;
    let lr = Constant(fp.quantization(0.2));
    let m_inv = Constant(fp.quantization(1.0 / (train_m as f64)));
    let one = Constant(fp.quantization(1.0));
    let half = Constant(fp.quantization(0.5));
    let quarter = Constant(fp.quantization(0.25));
    let one_over_48 = Constant(fp.quantization(1.0 / 48.0));
    let zero = ctx.load_constant(fp.quantization(0.0));

    // Load data
    let mut tr_x: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..train_m {
        let xi0 = ctx.load_witness(fp.quantization(input.training_x[i][0]));
        let xi1 = ctx.load_witness(fp.quantization(input.training_x[i][1]));
        tr_x.push(vec![xi0, xi1]);
    }
    let tr_y: Vec<AssignedValue<F>> = (0..train_m)
        .map(|i| ctx.load_witness(fp.quantization(input.training_y[i])))
        .collect();

    let mut te_x: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..test_m {
        let xi0 = ctx.load_witness(fp.quantization(input.testing_x[i][0]));
        let xi1 = ctx.load_witness(fp.quantization(input.testing_x[i][1]));
        te_x.push(vec![xi0, xi1]);
    }
    let te_y: Vec<AssignedValue<F>> = (0..test_m)
        .map(|i| ctx.load_witness(fp.quantization(input.testing_y[i])))
        .collect();

    // Parameters
    let mut w0 = zero;
    let mut w1 = zero;
    let mut b = zero;

    // Training loop
    for _ in 0..100 {
        // compute z_i = w·x + b
        let mut preds: Vec<AssignedValue<F>> = Vec::new();
        for i in 0..train_m {
            let s0 = fp.qmul(ctx, tr_x[i][0], w0);
            let s1 = fp.qmul(ctx, tr_x[i][1], w1);
            let tmp = fp.qadd(ctx, s0, s1);
            let z = fp.qadd(ctx, tmp, b);

            // sigmoid(z) ≈ 0.5 + 0.25z - (z^3)/48
            let z2 = fp.qmul(ctx, z, z);
            let z3 = fp.qmul(ctx, z2, z);
            let term1 = fp.qmul(ctx, quarter, z);
            let term2 = fp.qmul(ctx, one_over_48, z3);
            let t1_sub_t2 = fp.qsub(ctx, term1, term2);
            let sig = fp.qadd(ctx, half, t1_sub_t2);
            preds.push(sig);
        }

        // errors = preds - y
        let mut errors: Vec<AssignedValue<F>> = Vec::new();
        for i in 0..train_m {
            let e = fp.qsub(ctx, preds[i], tr_y[i]);
            errors.push(e);
        }

        // dw0, dw1, db
        let mut dw0 = zero;
        let mut dw1 = zero;
        let mut db = zero;
        for i in 0..train_m {
            let tmp1 = fp.qmul(ctx, tr_x[i][0], errors[i]);
            let tmp2 = fp.qmul(ctx, tr_x[i][1], errors[i]);
            dw0 = fp.qadd(ctx, dw0, tmp1);
            dw1 = fp.qadd(ctx, dw1, tmp2);
            db = fp.qadd(ctx, db, errors[i]);
        }

        // average
        dw0 = fp.qmul(ctx, dw0, m_inv);
        dw1 = fp.qmul(ctx, dw1, m_inv);
        db = fp.qmul(ctx, db, m_inv);

        // parameter update
        let lr_dw0 = fp.qmul(ctx, lr, dw0);
        let lr_dw1 = fp.qmul(ctx, lr, dw1);
        let lr_db = fp.qmul(ctx, lr, db);
        w0 = fp.qsub(ctx, w0, lr_dw0);
        w1 = fp.qsub(ctx, w1, lr_dw1);
        b = fp.qsub(ctx, b, lr_db);
    }

    // Evaluation
    let mut mismatches = zero;
    for i in 0..test_m {
        let s0 = fp.qmul(ctx, te_x[i][0], w0);
        let s1 = fp.qmul(ctx, te_x[i][1], w1);
        let tmp = fp.qadd(ctx, s0, s1);
        let z = fp.qadd(ctx, tmp, b);

        // sigmoid approximation again
        let z2 = fp.qmul(ctx, z, z);
        let z3 = fp.qmul(ctx, z2, z);
        let term1 = fp.qmul(ctx, quarter, z);
        let term2 = fp.qmul(ctx, one_over_48, z3);
        let t1_sub_t2 = fp.qsub(ctx, term1, term2);
        let prob = fp.qadd(ctx, half, t1_sub_t2);

        // pred = 1 if prob >= 0.5 else 0
        let lt_half = range.is_less_than(ctx, prob, half, 128);
        let ge_half = gate.not(ctx, lt_half);
        let pred = gate.select(ctx, one, Constant(fp.quantization(0.0)), ge_half);

        // mismatch += (pred != y)
        let diff = fp.qsub(ctx, pred, te_y[i]);
        let abs_diff = {
            let neg_diff = fp.neg(ctx, diff);
            let lt0 = range.is_less_than(ctx, diff, zero, 128);
            gate.select(ctx, neg_diff, diff, lt0)
        };
        mismatches = fp.qadd(ctx, mismatches, abs_diff);
    }

    // Require mismatches == 0
    let ok = gate.is_equal(ctx, mismatches, zero);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
