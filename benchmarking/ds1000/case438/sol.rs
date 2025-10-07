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
    pub post: Vec<f64>,       // len = 4
    pub distance: Vec<f64>,   // len = 4
    pub result: f64,          // scalar Pearson r
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

    // Load inputs
    let post: Vec<AssignedValue<F>> = input
        .post
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();
    let distance: Vec<AssignedValue<F>> = input
        .distance
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();
    let out_r = ctx.load_witness(fp.quantization(input.result));

    // Constants
    let n = 4usize;
    let n_f = Constant(fp.quantization(n as f64));
    let zero = ctx.load_constant(fp.quantization(0.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));

    // mean_post = sum(post)/n
    let mut sum_post = zero;
    for i in 0..n {
        sum_post = fp.qadd(ctx, sum_post, post[i]);
    }
    let mean_post = fp.qdiv(ctx, sum_post, n_f);

    // mean_distance = sum(distance)/n
    let mut sum_dist = zero;
    for i in 0..n {
        sum_dist = fp.qadd(ctx, sum_dist, distance[i]);
    }
    let mean_distance = fp.qdiv(ctx, sum_dist, n_f);

    // cov = sum((post[i]-mean_post)*(distance[i]-mean_distance)) / n
    let mut cov_sum = zero;
    for i in 0..n {
        let dp = fp.qsub(ctx, post[i], mean_post);
        let dd = fp.qsub(ctx, distance[i], mean_distance);
        let prod = fp.qmul(ctx, dp, dd);
        cov_sum = fp.qadd(ctx, cov_sum, prod);
    }
    let cov = fp.qdiv(ctx, cov_sum, n_f);

    // var_post and var_distance (population)
    let mut sse_post = zero;
    let mut sse_dist = zero;
    for i in 0..n {
        let dp = fp.qsub(ctx, post[i], mean_post);
        let dd = fp.qsub(ctx, distance[i], mean_distance);
    
        let dp_sq = fp.qmul(ctx, dp, dp);
        sse_post = fp.qadd(ctx, sse_post, dp_sq);
    
        let dd_sq = fp.qmul(ctx, dd, dd);
        sse_dist = fp.qadd(ctx, sse_dist, dd_sq);
    }
    let var_post = fp.qdiv(ctx, sse_post, n_f);
    let var_distance = fp.qdiv(ctx, sse_dist, n_f);

    // stds
    let std_post = fp.qsqrt(ctx, var_post);
    let std_distance = fp.qsqrt(ctx, var_distance);

    // pearson r = cov / (std_post * std_distance)
    let denom = fp.qmul(ctx, std_post, std_distance);
    let r = fp.qdiv(ctx, cov, denom);

    // Assert result == r within ±1e-3
    let diff = fp.qsub(ctx, out_r, r);
    let le = range.is_less_than(ctx, diff, tol_pos, 128);
    let ge = range.is_less_than(ctx, tol_neg, diff, 128);
    let ok = gate.and(ctx, le, ge);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
