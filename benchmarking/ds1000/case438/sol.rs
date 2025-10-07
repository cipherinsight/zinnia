use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
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
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub post: Vec<f64>,
    pub distance: Vec<f64>,
    pub result: f64,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range = builder.range_chip();
    let fixed = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());

    let n = input.post.len() as f64;
    let n_const = Constant(fixed.quantization(n));

    // --- Load inputs ---
    let post: Vec<AssignedValue<F>> = input
        .post
        .iter()
        .map(|x| ctx.load_witness(fixed.quantization(*x)))
        .collect();
    let distance: Vec<AssignedValue<F>> = input
        .distance
        .iter()
        .map(|x| ctx.load_witness(fixed.quantization(*x)))
        .collect();
    let result = ctx.load_witness(fixed.quantization(input.result));

    // --- Step 1: means ---
    let mut sum_post = ctx.load_constant(fixed.quantization(0.0));
    let mut sum_dist = ctx.load_constant(fixed.quantization(0.0));
    for i in 0..post.len() {
        sum_post = fixed.qadd(ctx, sum_post, post[i]);
        sum_dist = fixed.qadd(ctx, sum_dist, distance[i]);
    }
    let mean_post = fixed.qdiv(ctx, sum_post, n_const);
    let mean_dist = fixed.qdiv(ctx, sum_dist, n_const);

    // --- Step 2: covariance ---
    let mut cov_sum = ctx.load_constant(fixed.quantization(0.0));
    for i in 0..post.len() {
        let dp = fixed.qsub(ctx, post[i], mean_post);
        let dd = fixed.qsub(ctx, distance[i], mean_dist);
        let prod = fixed.qmul(ctx, dp, dd);
        cov_sum = fixed.qadd(ctx, cov_sum, prod);
    }
    let cov = fixed.qdiv(ctx, cov_sum, n_const);

    // --- Step 3: variances ---
    let mut var_post_sum = ctx.load_constant(fixed.quantization(0.0));
    let mut var_dist_sum = ctx.load_constant(fixed.quantization(0.0));
    for i in 0..post.len() {
        let dp = fixed.qsub(ctx, post[i], mean_post);
        let dd = fixed.qsub(ctx, distance[i], mean_dist);
        let dp_sq = fixed.qmul(ctx, dp, dp);
        let dd_sq = fixed.qmul(ctx, dd, dd);
        var_post_sum = fixed.qadd(ctx, var_post_sum, dp_sq);
        var_dist_sum = fixed.qadd(ctx, var_dist_sum, dd_sq);
    }
    let var_post = fixed.qdiv(ctx, var_post_sum, n_const);
    let var_dist = fixed.qdiv(ctx, var_dist_sum, n_const);

    // --- Step 4: standard deviations ---
    let std_post = fixed.qsqrt(ctx, var_post);
    let std_dist = fixed.qsqrt(ctx, var_dist);

    // --- Step 5: correlation ---
    let denom = fixed.qmul(ctx, std_post, std_dist);
    let pearson_r = fixed.qdiv(ctx, cov, denom);

    // --- Step 6: assert equality ---
    let diff = fixed.qsub(ctx, pearson_r, result);
    let within_upper = range.is_less_than(ctx, diff, Constant(fixed.quantization(0.001)), 128);
    let within_lower = range.is_less_than(ctx, Constant(fixed.quantization(-0.001)), diff, 128);
    let eq = gate.and(ctx, within_upper, within_lower);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
