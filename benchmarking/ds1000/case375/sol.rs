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
    pub grades: Vec<f64>,  // len = 27, already sorted (verified in-circuit)
    pub threshold: f64,
    pub low: f64,
    pub high: f64,
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

    let n = 27usize;

    // Load inputs
    let grades: Vec<AssignedValue<F>> = input
        .grades
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();
    let threshold = ctx.load_witness(fp.quantization(input.threshold));
    let out_low = ctx.load_witness(fp.quantization(input.low));
    let out_high = ctx.load_witness(fp.quantization(input.high));

    // Tolerance constants
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));

    // 1) Verify sortedness (non-decreasing): grades[i] <= grades[i+1]
    for i in 0..(n - 1) {
        // grades[i+1] < grades[i] must be false
        let bad = range.is_less_than(ctx, grades[i + 1], grades[i], 128);
        let ok = gate.not(ctx, bad);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }

    // 2) Find first index t where (k+1)/n > threshold
    let n_f = ctx.load_constant(fp.quantization(n as f64));
    let mut t = ctx.load_constant(fp.quantization(n as f64)); // sentinel t = n
    for k in 0..n {
        let k1 = Constant(fp.quantization((k + 1) as f64));
        let frac = fp.qdiv(ctx, k1, n_f); // (k+1)/n

        // cond: frac > threshold  <=>  threshold < frac
        let cond_gt = range.is_less_than(ctx, threshold, frac, 128);

        // update only if we haven't set t yet (t == n)
        let t_is_n = gate.is_equal(ctx, t, Constant(fp.quantization(n as f64)));
        let cond = gate.and(ctx, cond_gt, t_is_n);

        // t = cond ? k : t
        let k_const = ctx.load_constant(fp.quantization(k as f64));
        t = gate.select(ctx, k_const, t, cond);
    }

    // 3) computed_low = grades[0]; computed_high = grades[t]
    let computed_low = grades[0];

    // Select grades[t]
    let mut computed_high = ctx.load_constant(fp.quantization(0.0));
    for k in 0..n {
        let k_const = Constant(fp.quantization(k as f64));
        let is_k = gate.is_equal(ctx, t, k_const);
        computed_high = gate.select(ctx, grades[k], computed_high, is_k);
    }

    // 4) Verify outputs within ±1e-3
    // low
    let d_low = fp.qsub(ctx, out_low, computed_low);
    let low_ok_hi = range.is_less_than(ctx, d_low, tol_pos, 128);
    let low_ok_lo = range.is_less_than(ctx, tol_neg, d_low, 128);
    let low_ok = gate.and(ctx, low_ok_hi, low_ok_lo);
    gate.assert_is_const(ctx, &low_ok, &F::ONE);

    // high
    let d_high = fp.qsub(ctx, out_high, computed_high);
    let high_ok_hi = range.is_less_than(ctx, d_high, tol_pos, 128);
    let high_ok_lo = range.is_less_than(ctx, tol_neg, d_high, 128);
    let high_ok = gate.and(ctx, high_ok_hi, high_ok_lo);
    gate.assert_is_const(ctx, &high_ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
