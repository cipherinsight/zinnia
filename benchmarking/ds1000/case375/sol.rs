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
use halo2_graph::gadget::fixed_point::FixedPointChip;
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
    pub grades: Vec<f64>,
    pub threshold: f64,
    pub low: f64,
    pub high: f64,
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
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let n = input.grades.len();

    // ---- Step 1: load all witnesses ----
    let mut grades_fp: Vec<AssignedValue<F>> = Vec::new();
    for g in &input.grades {
        grades_fp.push(ctx.load_witness(fixed_point_chip.quantization(*g)));
    }

    let threshold_fp = ctx.load_witness(fixed_point_chip.quantization(input.threshold));
    let low_fp = ctx.load_witness(fixed_point_chip.quantization(input.low));
    let high_fp = ctx.load_witness(fixed_point_chip.quantization(input.high));

    // ---- Step 2: verify sortedness (non-decreasing) ----
    for i in 0..(n - 1) {
        let lt = range_chip.is_less_than(ctx, grades_fp[i], grades_fp[i + 1], 128);
        let eq = gate.is_equal(ctx, grades_fp[i], grades_fp[i + 1]);
        let le_or_eq = gate.or(ctx, lt, eq);
        gate.assert_is_const(ctx, &le_or_eq, &F::ONE);
    }

    // ---- Step 3: find t = smallest k such that (k+1)/n > threshold ----
    let mut t_flags: Vec<AssignedValue<F>> = Vec::new();
    let mut found_any = ctx.load_constant(F::ZERO);

    for k in 0..n {
        let val = fixed_point_chip.quantization((k as f64 + 1.0) / n as f64);
        let yk = ctx.load_constant(val);
        let gt = range_chip.is_less_than(ctx, threshold_fp, yk, 128); // threshold < yk → yk > threshold
        let not_found = gate.not(ctx, found_any);
        let first_hit = gate.and(ctx, gt, not_found);
        found_any = gate.or(ctx, found_any, gt);
        t_flags.push(first_hit);
    }

    // ---- Step 4: derive computed_low, computed_high ----
    let computed_low = grades_fp[0];
    let mut computed_high = ctx.load_constant(fixed_point_chip.quantization(0.0));
    for k in 0..n {
        computed_high = gate.select(ctx, grades_fp[k], computed_high, t_flags[k]);
    }

    // ---- Step 5: compare with provided outputs ----
    let eq_low = gate.is_equal(ctx, computed_low, low_fp);
    let eq_high = gate.is_equal(ctx, computed_high, high_fp);
    let both = gate.and(ctx, eq_low, eq_high);
    gate.assert_is_const(ctx, &both, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
