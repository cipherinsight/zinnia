use std::result;
use clap::Parser;
use halo2_base::utils::{BigPrimeField, ScalarField};
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
use serde::{Deserialize, Serialize};
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

    // Shapes / hyperparams
    let input_dim = 2usize;
    let hidden_dim = 3usize;
    let train_m = 10usize;
    let test_m = 2usize;
    let steps = 10usize;

    // Constants
    let lr = Constant(fp.quantization(0.02));
    let inv_m = Constant(fp.quantization(1.0 / (train_m as f64)));
    let two = Constant(fp.quantization(2.0));
    let four = Constant(fp.quantization(4.0));
    let zero = ctx.load_constant(fp.quantization(0.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));
    let fifty = Constant(fp.quantization(50.0));

    // inv_m2 = 2 / m
    let inv_m2 = fp.qmul(ctx, two, inv_m);
    // (4/m)
    let four_over_m = fp.qmul(ctx, four, inv_m);

    // Load data
    let mut train_x: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(train_m);
    for i in 0..train_m {
        let mut row = Vec::with_capacity(input_dim);
        for j in 0..input_dim {
            row.push(ctx.load_witness(fp.quantization(input.training_x[i][j])));
        }
        train_x.push(row);
    }
    let train_y: Vec<AssignedValue<F>> = (0..train_m)
        .map(|i| ctx.load_witness(fp.quantization(input.training_y[i])))
        .collect();

    let mut test_x: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(test_m);
    for i in 0..test_m {
        let mut row = Vec::with_capacity(input_dim);
        for j in 0..input_dim {
            row.push(ctx.load_witness(fp.quantization(input.testing_x[i][j])));
        }
        test_x.push(row);
    }
    let test_y: Vec<AssignedValue<F>> = (0..test_m)
        .map(|i| ctx.load_witness(fp.quantization(input.testing_y[i])))
        .collect();

    // Initialize parameters
    // W1: 2 x 3
    let mut W1 = vec![
        vec![
            ctx.load_constant(fp.quantization(0.02)),
            ctx.load_constant(fp.quantization(-0.01)),
            ctx.load_constant(fp.quantization(0.016)),
        ],
        vec![
            ctx.load_constant(fp.quantization(0.014)),
            ctx.load_constant(fp.quantization(0.004)),
            ctx.load_constant(fp.quantization(-0.006)),
        ],
    ];
    // b1: 3
    let mut b1 = vec![zero, zero, zero];
    // W2: 3
    let mut W2 = vec![
        ctx.load_constant(fp.quantization(0.05)),
        ctx.load_constant(fp.quantization(-0.03)),
        ctx.load_constant(fp.quantization(0.02)),
    ];
    // b2: scalar
    let mut b2 = zero;

    // Training loop
    for _ in 0..steps {
        // Forward pass (training)
        let mut H: Vec<Vec<AssignedValue<F>>> = vec![vec![zero; hidden_dim]; train_m];
        let mut A: Vec<Vec<AssignedValue<F>>> = vec![vec![zero; hidden_dim]; train_m];
        let mut preds: Vec<AssignedValue<F>> = vec![zero; train_m];

        for i in 0..train_m {
            // For each hidden unit k: h = x·W1[:,k] + b1[k]; a = h^2
            for k in 0..hidden_dim {
                // h = x0*W1[0][k] + x1*W1[1][k] + b1[k]
                let x0w = fp.qmul(ctx, train_x[i][0], W1[0][k]);
                let x1w = fp.qmul(ctx, train_x[i][1], W1[1][k]);
                let tmp = fp.qadd(ctx, x0w, x1w);
                let h = fp.qadd(ctx, tmp, b1[k]);
                H[i][k] = h;

                let h2 = fp.qmul(ctx, h, h);
                A[i][k] = h2;
            }
            // out = A[i]·W2 + b2
            let a0w = fp.qmul(ctx, A[i][0], W2[0]);
            let a1w = fp.qmul(ctx, A[i][1], W2[1]);
            let a2w = fp.qmul(ctx, A[i][2], W2[2]);
            let t01 = fp.qadd(ctx, a0w, a1w);
            let t012 = fp.qadd(ctx, t01, a2w);
            let out = fp.qadd(ctx, t012, b2);
            preds[i] = out;
        }

        // errors = preds - training_y
        let mut errors: Vec<AssignedValue<F>> = vec![zero; train_m];
        for i in 0..train_m {
            errors[i] = fp.qsub(ctx, preds[i], train_y[i]);
        }

        // dW2, db2
        let mut dW2 = vec![zero, zero, zero];
        let mut db2 = zero;
        for i in 0..train_m {
            let e = errors[i];
            db2 = fp.qadd(ctx, db2, e);

            let e_a0 = fp.qmul(ctx, e, A[i][0]);
            dW2[0] = fp.qadd(ctx, dW2[0], e_a0);

            let e_a1 = fp.qmul(ctx, e, A[i][1]);
            dW2[1] = fp.qadd(ctx, dW2[1], e_a1);

            let e_a2 = fp.qmul(ctx, e, A[i][2]);
            dW2[2] = fp.qadd(ctx, dW2[2], e_a2);
        }
        dW2[0] = fp.qmul(ctx, dW2[0], inv_m2);
        dW2[1] = fp.qmul(ctx, dW2[1], inv_m2);
        dW2[2] = fp.qmul(ctx, dW2[2], inv_m2);
        db2 = fp.qmul(ctx, db2, inv_m2);

        // Hidden layer grads via chain rule; dφ/dh = 2h
        // dh = (4/m) * e * W2[k] * H[i][k]
        let mut dW1 = vec![vec![zero; hidden_dim]; input_dim]; // [2][3]
        let mut db1g = vec![zero, zero, zero];
        for i in 0..train_m {
            let e = errors[i];
            for k in 0..hidden_dim {
                let e_w = fp.qmul(ctx, e, W2[k]);
                let e_w_h = fp.qmul(ctx, e_w, H[i][k]);
                let dh = fp.qmul(ctx, four_over_m, e_w_h);

                // dW1[:,k] += dh * x
                let dh_x0 = fp.qmul(ctx, dh, train_x[i][0]);
                dW1[0][k] = fp.qadd(ctx, dW1[0][k], dh_x0);

                let dh_x1 = fp.qmul(ctx, dh, train_x[i][1]);
                dW1[1][k] = fp.qadd(ctx, dW1[1][k], dh_x1);

                // db1g[k] += dh
                db1g[k] = fp.qadd(ctx, db1g[k], dh);
            }
        }

        // Parameter updates
        for k in 0..hidden_dim {
            // W2[k] -= lr * dW2[k]
            let lr_dw2 = fp.qmul(ctx, lr, dW2[k]);
            W2[k] = fp.qsub(ctx, W2[k], lr_dw2);

            // b1[k] -= lr * db1g[k]
            let lr_db1 = fp.qmul(ctx, lr, db1g[k]);
            b1[k] = fp.qsub(ctx, b1[k], lr_db1);

            // W1[0][k] -= lr * dW1[0][k]
            let lr_dw10 = fp.qmul(ctx, lr, dW1[0][k]);
            W1[0][k] = fp.qsub(ctx, W1[0][k], lr_dw10);

            // W1[1][k] -= lr * dW1[1][k]
            let lr_dw11 = fp.qmul(ctx, lr, dW1[1][k]);
            W1[1][k] = fp.qsub(ctx, W1[1][k], lr_dw11);
        }
        // b2 -= lr * db2
        let lr_db2 = fp.qmul(ctx, lr, db2);
        b2 = fp.qsub(ctx, b2, lr_db2);
    }

    // Forward pass on testing set
    let mut test_preds = vec![zero; test_m];
    for i in 0..test_m {
        // hidden pre-acts
        // k=0
        let x0w00 = fp.qmul(ctx, test_x[i][0], W1[0][0]);
        let x1w10 = fp.qmul(ctx, test_x[i][1], W1[1][0]);
        let s00 = fp.qadd(ctx, x0w00, x1w10);
        let h0 = fp.qadd(ctx, s00, b1[0]);
        let a0 = fp.qmul(ctx, h0, h0);

        // k=1
        let x0w01 = fp.qmul(ctx, test_x[i][0], W1[0][1]);
        let x1w11 = fp.qmul(ctx, test_x[i][1], W1[1][1]);
        let s01 = fp.qadd(ctx, x0w01, x1w11);
        let h1 = fp.qadd(ctx, s01, b1[1]);
        let a1 = fp.qmul(ctx, h1, h1);

        // k=2
        let x0w02 = fp.qmul(ctx, test_x[i][0], W1[0][2]);
        let x1w12 = fp.qmul(ctx, test_x[i][1], W1[1][2]);
        let s02 = fp.qadd(ctx, x0w02, x1w12);
        let h2 = fp.qadd(ctx, s02, b1[2]);
        let a2 = fp.qmul(ctx, h2, h2);

        // yhat = a·W2 + b2
        let a0w = fp.qmul(ctx, a0, W2[0]);
        let a1w = fp.qmul(ctx, a1, W2[1]);
        let a2w = fp.qmul(ctx, a2, W2[2]);
        let t01 = fp.qadd(ctx, a0w, a1w);
        let t012 = fp.qadd(ctx, t01, a2w);
        let yhat = fp.qadd(ctx, t012, b2);
        test_preds[i] = yhat;
    }

    // Test MSE
    let mut se_sum = zero;
    for i in 0..test_m {
        let diff = fp.qsub(ctx, test_preds[i], test_y[i]);
        let diff2 = fp.qmul(ctx, diff, diff);
        se_sum = fp.qadd(ctx, se_sum, diff2);
    }
    let test_m_f = Constant(fp.quantization(test_m as f64));
    let test_mse = fp.qdiv(ctx, se_sum, test_m_f);

    // Assert test_mse <= 50  (LE_F via: !(50 < test_mse))
    let fifty_lt_mse = range.is_less_than(ctx, fifty, test_mse, 128);
    let ok = gate.not(ctx, fifty_lt_mse);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
