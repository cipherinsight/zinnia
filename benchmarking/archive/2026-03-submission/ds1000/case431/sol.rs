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
    pub a: Vec<f64>,     // len = 13 (floats)
    pub result: Vec<u64> // len = 13 (ints 0/1)
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
    let res: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // Constants
    let n = 13usize;
    let n_f = Constant(fp.quantization(n as f64));
    let two = Constant(fp.quantization(2.0));
    let zero = ctx.load_constant(fp.quantization(0.0));

    // mean = sum(a)/n
    let mut sum = zero;
    for i in 0..n {
        sum = fp.qadd(ctx, sum, a[i]);
    }
    let mean = fp.qdiv(ctx, sum, n_f);

    // variance = sum((a[i]-mean)^2)/n
    let mut sse = zero;
    for i in 0..n {
        let diff = fp.qsub(ctx, a[i], mean);
        let sq = fp.qmul(ctx, diff, diff);
        sse = fp.qadd(ctx, sse, sq);
    }
    let variance = fp.qdiv(ctx, sse, n_f);

    // std = sqrt(variance)
    let std = fp.qsqrt(ctx, variance);

    // bounds: lower = mean - 2*std; upper = mean + 2*std
    let two_std = fp.qmul(ctx, two, std);
    let lower = fp.qsub(ctx, mean, two_std);
    let upper = fp.qadd(ctx, mean, two_std);

    // For each i:
    //   inside = (a[i] > lower) AND (a[i] < upper)
    //   expected = NOT inside  (1 if outside, else 0)
    //   assert result[i] == expected
    for i in 0..n {
        // a[i] > lower  <=>  lower < a[i]
        let gt_lower = range.is_less_than(ctx, lower, a[i], 128);
        // a[i] < upper
        let lt_upper = range.is_less_than(ctx, a[i], upper, 128);
        let inside = gate.and(ctx, gt_lower, lt_upper);
        let expected_bool = gate.not(ctx, inside);
        let expected_val = gate.select(ctx, Constant(F::ONE), Constant(F::ZERO), expected_bool);

        let eq = gate.is_equal(ctx, res[i], expected_val);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
