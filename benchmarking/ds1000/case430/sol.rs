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
    pub a: Vec<f64>,      // len = 13
    pub result: Vec<f64>, // len = 2 -> [lower, upper]
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

    // Load inputs
    let a: Vec<AssignedValue<F>> = input
        .a
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();
    let out: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();

    // Constants
    let n = 13usize;
    let n_f = Constant(fp.quantization(n as f64));
    let three = Constant(fp.quantization(3.0));
    let zero = ctx.load_constant(fp.quantization(0.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));

    // mean = sum(a) / n
    let mut sum = zero;
    for i in 0..n {
        sum = fp.qadd(ctx, sum, a[i]);
    }
    let mean = fp.qdiv(ctx, sum, n_f);

    // variance = sum((a[i] - mean)^2) / n
    let mut sse = zero;
    for i in 0..n {
        let diff = fp.qsub(ctx, a[i], mean);
        let sq = fp.qmul(ctx, diff, diff);
        sse = fp.qadd(ctx, sse, sq);
    }
    let variance = fp.qdiv(ctx, sse, n_f);

    // std = sqrt(variance)
    let std = fp.qsqrt(ctx, variance);

    // lower = mean - 3*std; upper = mean + 3*std
    let three_std = fp.qmul(ctx, three, std);
    let lower = fp.qsub(ctx, mean, three_std);
    let upper = fp.qadd(ctx, mean, three_std);

    // Assert result == (lower, upper) within ±1e-3
    let d0 = fp.qsub(ctx, out[0], lower);
    let ok0_lo = range.is_less_than(ctx, d0, tol_pos, 128);
    let ok0_hi = range.is_less_than(ctx, tol_neg, d0, 128);
    let ok0 = gate.and(ctx, ok0_lo, ok0_hi);
    gate.assert_is_const(ctx, &ok0, &F::ONE);

    let d1 = fp.qsub(ctx, out[1], upper);
    let ok1_lo = range.is_less_than(ctx, d1, tol_pos, 128);
    let ok1_hi = range.is_less_than(ctx, tol_neg, d1, 128);
    let ok1 = gate.and(ctx, ok1_lo, ok1_hi);
    gate.assert_is_const(ctx, &ok1, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
