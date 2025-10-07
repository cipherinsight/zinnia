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
    pub training_x: Vec<Vec<f64>>, // 10 x 2 (binary features in {0,1})
    pub training_y: Vec<f64>,      // 10 (labels in {0,1})
    pub testing_x: Vec<Vec<f64>>,  // 2 x 2
    pub testing_y: Vec<f64>,       // 2 (labels in {0,1})
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
    let n_train = 10usize;
    let n_classes = 2usize;
    let half = Constant(fp.quantization(0.5));
    let one = Constant(fp.quantization(1.0));
    let two = Constant(fp.quantization(2.0));
    let alpha = Constant(fp.quantization(1.0));
    let zero = ctx.load_constant(fp.quantization(0.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));

    // Load data
    let mut tr_x: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(n_train);
    for i in 0..n_train {
        let x0 = ctx.load_witness(fp.quantization(input.training_x[i][0]));
        let x1 = ctx.load_witness(fp.quantization(input.training_x[i][1]));
        tr_x.push(vec![x0, x1]);
    }
    let tr_y: Vec<AssignedValue<F>> = (0..n_train)
        .map(|i| ctx.load_witness(fp.quantization(input.training_y[i])))
        .collect();

    let mut te_x: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(2);
    for i in 0..2 {
        let x0 = ctx.load_witness(fp.quantization(input.testing_x[i][0]));
        let x1 = ctx.load_witness(fp.quantization(input.testing_x[i][1]));
        te_x.push(vec![x0, x1]);
    }
    let te_y: Vec<AssignedValue<F>> = (0..2)
        .map(|i| ctx.load_witness(fp.quantization(input.testing_y[i])))
        .collect();

    // Counters
    let mut count_c0 = zero; // #examples with y=0
    let mut count_c1 = zero; // #examples with y=1
    let mut count1_c0_f0 = zero; // feature 0 = 1 given class 0
    let mut count1_c0_f1 = zero; // feature 1 = 1 given class 0
    let mut count1_c1_f0 = zero; // feature 0 = 1 given class 1
    let mut count1_c1_f1 = zero; // feature 1 = 1 given class 1

    for i in 0..n_train {
        let yi = tr_y[i];
        let x0 = tr_x[i][0];
        let x1 = tr_x[i][1];

        // yi == 0.0 ?
        let yi_is_zero = gate.is_equal(ctx, yi, Constant(fp.quantization(0.0)));
        let yi_is_one  = gate.is_equal(ctx, yi, one);

        // xj >= 0.5  <=>  !(xj < 0.5)
        let x0_lt_half = range.is_less_than(ctx, x0, half, 128);
        let x1_lt_half = range.is_less_than(ctx, x1, half, 128);
        let x0_ge_half = gate.not(ctx, x0_lt_half);
        let x1_ge_half = gate.not(ctx, x1_lt_half);

        // Increment class counts
        let c0_plus = fp.qadd(ctx, count_c0, one);
        count_c0 = gate.select(ctx, c0_plus, count_c0, yi_is_zero);

        let c1_plus = fp.qadd(ctx, count_c1, one);
        count_c1 = gate.select(ctx, c1_plus, count_c1, yi_is_one);

        // For class 0 feature counts: increment if yi==0 && xj>=0.5
        let inc_c0_f0 = gate.and(ctx, yi_is_zero, x0_ge_half);
        let inc_c0_f1 = gate.and(ctx, yi_is_zero, x1_ge_half);

        let c0f0_plus = fp.qadd(ctx, count1_c0_f0, one);
        count1_c0_f0 = gate.select(ctx, c0f0_plus, count1_c0_f0, inc_c0_f0);

        let c0f1_plus = fp.qadd(ctx, count1_c0_f1, one);
        count1_c0_f1 = gate.select(ctx, c0f1_plus, count1_c0_f1, inc_c0_f1);

        // For class 1 feature counts: increment if yi==1 && xj>=0.5
        let inc_c1_f0 = gate.and(ctx, yi_is_one, x0_ge_half);
        let inc_c1_f1 = gate.and(ctx, yi_is_one, x1_ge_half);

        let c1f0_plus = fp.qadd(ctx, count1_c1_f0, one);
        count1_c1_f0 = gate.select(ctx, c1f0_plus, count1_c1_f0, inc_c1_f0);

        let c1f1_plus = fp.qadd(ctx, count1_c1_f1, one);
        count1_c1_f1 = gate.select(ctx, c1f1_plus, count1_c1_f1, inc_c1_f1);
    }

    // Priors with Laplace smoothing:
    // prior_c = (count_c + alpha) / (n_train + n_classes * alpha)
    let n_train_f = Constant(fp.quantization(n_train as f64));
    let ncls_alpha = fp.qmul(ctx, Constant(fp.quantization(n_classes as f64)), alpha);
    let denom_prior_sum = fp.qadd(ctx, n_train_f, ncls_alpha);

    let c0_pa = fp.qadd(ctx, count_c0, alpha);
    let prior0 = fp.qdiv(ctx, c0_pa, denom_prior_sum);

    let c1_pa = fp.qadd(ctx, count_c1, alpha);
    let prior1 = fp.qdiv(ctx, c1_pa, denom_prior_sum);

    // Bernoulli likelihoods with Laplace smoothing over {0,1}:
    // denom_c = count_c + 2*alpha
    let two_alpha = fp.qmul(ctx, two, alpha);
    let denom0 = fp.qadd(ctx, count_c0, two_alpha);
    let denom1 = fp.qadd(ctx, count_c1, two_alpha);

    let num0_f0 = fp.qadd(ctx, count1_c0_f0, alpha);
    let theta0_f0 = fp.qdiv(ctx, num0_f0, denom0);
    let num0_f1 = fp.qadd(ctx, count1_c0_f1, alpha);
    let theta0_f1 = fp.qdiv(ctx, num0_f1, denom0);

    let num1_f0 = fp.qadd(ctx, count1_c1_f0, alpha);
    let theta1_f0 = fp.qdiv(ctx, num1_f0, denom1);
    let num1_f1 = fp.qadd(ctx, count1_c1_f1, alpha);
    let theta1_f1 = fp.qdiv(ctx, num1_f1, denom1);

    // Predict on the 2 test points:
    // Score_c(x) = prior_c * Π_j [ θ_{c,j}^xj * (1-θ_{c,j})^(1-xj) ]
    // Since xj ∈ {0,1}, we implement term(j) = (xj>=0.5) ? θ : (1-θ)
    for i in 0..2 {
        let x0 = te_x[i][0];
        let x1 = te_x[i][1];
        let x0_ge = gate.not(ctx, range.is_less_than(ctx, x0, half, 128));
        let x1_ge = gate.not(ctx, range.is_less_than(ctx, x1, half, 128));

        // class 0 terms
        let one_minus_theta0_f0 = {
            let tmp = fp.qsub(ctx, one, theta0_f0);
            tmp
        };
        let t0_f0 = gate.select(ctx, theta0_f0, one_minus_theta0_f0, x0_ge);

        let one_minus_theta0_f1 = {
            let tmp = fp.qsub(ctx, one, theta0_f1);
            tmp
        };
        let t0_f1 = gate.select(ctx, theta0_f1, one_minus_theta0_f1, x1_ge);

        let t0_prod = {
            let p = fp.qmul(ctx, t0_f0, t0_f1);
            p
        };
        let score0 = fp.qmul(ctx, prior0, t0_prod);

        // class 1 terms
        let one_minus_theta1_f0 = {
            let tmp = fp.qsub(ctx, one, theta1_f0);
            tmp
        };
        let t1_f0 = gate.select(ctx, theta1_f0, one_minus_theta1_f0, x0_ge);

        let one_minus_theta1_f1 = {
            let tmp = fp.qsub(ctx, one, theta1_f1);
            tmp
        };
        let t1_f1 = gate.select(ctx, theta1_f1, one_minus_theta1_f1, x1_ge);

        let t1_prod = {
            let p = fp.qmul(ctx, t1_f0, t1_f1);
            p
        };
        let score1 = fp.qmul(ctx, prior1, t1_prod);

        // pred = 1.0 if score1 >= score0 else 0.0
        // score1 >= score0  <=>  !(score1 < score0)
        let s1_lt_s0 = range.is_less_than(ctx, score1, score0, 128);
        let ge = gate.not(ctx, s1_lt_s0);
        let pred = gate.select(ctx, one, Constant(fp.quantization(0.0)), ge);

        // Check equality against testing_y[i] within ±1e-3
        let diff = fp.qsub(ctx, pred, te_y[i]);
        let ok_hi = range.is_less_than(ctx, diff, tol_pos, 128);
        let ok_lo = range.is_less_than(ctx, tol_neg, diff, 128);
        let ok = gate.and(ctx, ok_hi, ok_lo);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
