use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Witness},
};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use serde::{Serialize, Deserialize};
use halo2_base::poseidon::hasher::PoseidonHasher;
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub training_x: Vec<Vec<f64>>,
    pub training_y: Vec<f64>,
    pub testing_x: Vec<Vec<f64>>,
    pub testing_y: Vec<f64>,
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

    let lr = Constant(fp.quantization(0.2));
    let m = Constant(fp.quantization(10.0));

    // Load data
    let mut x_train: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for row in input.training_x.iter() {
        x_train.push(row.iter().map(|v| ctx.load_witness(fp.quantization(*v))).collect());
    }
    let y_train: Vec<AssignedValue<F>> =
        input.training_y.iter().map(|v| ctx.load_witness(fp.quantization(*v))).collect();
    let mut x_test: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for row in input.testing_x.iter() {
        x_test.push(row.iter().map(|v| ctx.load_witness(fp.quantization(*v))).collect());
    }
    let y_test: Vec<AssignedValue<F>> =
        input.testing_y.iter().map(|v| ctx.load_witness(fp.quantization(*v))).collect();

    // Initialize weights and bias
    let mut w = vec![ctx.load_constant(fp.quantization(0.0)), ctx.load_constant(fp.quantization(0.0))];
    let mut b = ctx.load_constant(fp.quantization(0.0));

    // Gradient descent
    for _ in 0..100 {
        // predictions
        let mut preds = Vec::new();
        for i in 0..10 {
            let tmp1 = fp.qmul(ctx, w[0], x_train[i][0]);
            let tmp2 = fp.qmul(ctx, w[1], x_train[i][1]);
            let lin = fp.qadd(ctx, tmp1, tmp2);
            let z = fp.qadd(ctx, lin, b);
            // sigmoid(z) = 1 / (1 + exp(-z))
            let neg_z = fp.neg(ctx, z);
            let exp_neg_z = fp.qexp(ctx, neg_z);
            let denom = fp.qadd(ctx, Constant(fp.quantization(1.0)), exp_neg_z);
            let s = fp.qdiv(ctx, Constant(fp.quantization(1.0)), denom);
            preds.push(s);
        }

        // errors = preds - y
        let mut errors = Vec::new();
        for i in 0..10 {
            errors.push(fp.qsub(ctx, preds[i], y_train[i]));
        }

        // dw_j
        let mut dw = vec![ctx.load_constant(fp.quantization(0.0)), ctx.load_constant(fp.quantization(0.0))];
        for j in 0..2 {
            let mut s = ctx.load_constant(fp.quantization(0.0));
            for i in 0..10 {
                let tmp = fp.qmul(ctx, x_train[i][j], errors[i]);
                s = fp.qadd(ctx, s, tmp);
            }
            dw[j] = fp.qdiv(ctx, s, m);
        }

        // db
        let mut s = ctx.load_constant(fp.quantization(0.0));
        for i in 0..10 {
            s = fp.qadd(ctx, s, errors[i]);
        }
        let db = fp.qdiv(ctx, s, m);

        // update
        for j in 0..2 {
            let step = fp.qmul(ctx, lr, dw[j]);
            w[j] = fp.qsub(ctx, w[j], step);
        }
        let tmp = fp.qmul(ctx, lr, db);
        b = fp.qsub(ctx, b, tmp);
    }

    // Testing phase
    let mut mismatches = ctx.load_constant(fp.quantization(0.0));
    for i in 0..2 {
        let tmp1 = fp.qmul(ctx, w[0], x_test[i][0]);
        let tmp2 = fp.qmul(ctx, w[1], x_test[i][1]);
        let lin = fp.qadd(ctx, tmp1, tmp2);
        let z = fp.qadd(ctx, lin, b);
        let neg_z = fp.neg(ctx, z);
        let exp_neg_z = fp.qexp(ctx, neg_z);
        let denom = fp.qadd(ctx, Constant(fp.quantization(1.0)), exp_neg_z);
        let p = fp.qdiv(ctx, Constant(fp.quantization(1.0)), denom);

        let cond = range.is_less_than(ctx, Constant(fp.quantization(0.5)), p, 128);
        let pred = gate.select(ctx, Constant(fp.quantization(1.0)), Constant(fp.quantization(0.0)), cond);

        let tmp = gate.is_equal(ctx, pred, y_test[i]);
        let neq = gate.not(ctx, tmp);
        let inc = fp.qadd(ctx, mismatches, Constant(fp.quantization(1.0)));
        mismatches = gate.select(ctx, inc, mismatches, neq);
    }

    // Require mismatches == 0
    let zero = ctx.load_constant(fp.quantization(0.0));
    let ok = gate.is_equal(ctx, mismatches, zero);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
