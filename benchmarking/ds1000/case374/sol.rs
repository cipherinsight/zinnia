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
    pub eval: Vec<f64>,
    pub result: Vec<f64>,
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
    let m = input.eval.len();

    // ---- Step 1: Load witnesses ----
    let mut grades_fp: Vec<AssignedValue<F>> = Vec::new();
    for g in &input.grades {
        grades_fp.push(ctx.load_witness(fixed_point_chip.quantization(*g)));
    }

    let mut eval_fp: Vec<AssignedValue<F>> = Vec::new();
    for e in &input.eval {
        eval_fp.push(ctx.load_witness(fixed_point_chip.quantization(*e)));
    }

    let mut result_fp: Vec<AssignedValue<F>> = Vec::new();
    for r in &input.result {
        result_fp.push(ctx.load_witness(fixed_point_chip.quantization(*r)));
    }

    // ---- Step 2: verify sortedness of grades ----
    for i in 0..(n - 1) {
        let lt = range_chip.is_less_than(ctx, grades_fp[i], grades_fp[i + 1], 128);
        let eq = gate.is_equal(ctx, grades_fp[i], grades_fp[i + 1]);
        let le_or_eq = gate.or(ctx, lt, eq);
        gate.assert_is_const(ctx, &le_or_eq, &F::ONE);
    }

    // ---- Step 3: compute ECDF values (ys[i] = (i+1)/n) ----
    let mut ys: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..n {
        let ideal = fixed_point_chip.quantization((i as f64 + 1.0) / n as f64);
        ys.push(ctx.load_constant(ideal));
    }

    // ---- Step 4: evaluate ECDF for each eval[i] ----
    for i in 0..m {
        let x = eval_fp[i];
        let mut computed = ctx.load_constant(fixed_point_chip.quantization(0.0));

        // Case 1: x < grades[0] -> 0.0
        let lt_min = range_chip.is_less_than(ctx, x, grades_fp[0], 128);

        // Case 2: x >= grades[n-1] -> 1.0
        let lt_max = range_chip.is_less_than(ctx, x, grades_fp[n - 1], 128);
        let ge_max = gate.not(ctx, lt_max);

        // Case 3: otherwise, find smallest j such that grades[j] > x
        // emulate sequential selection: computed = ys[j-1] when grades[j] > x and grades[j-1] <= x
        for j in 1..n {
            let gt = range_chip.is_less_than(ctx, x, grades_fp[j], 128); // grades[j] > x
            let le_prev = range_chip.is_less_than(ctx, grades_fp[j - 1], x, 128);
            let le_prev_not = gate.not(ctx, le_prev);
            let cond_window = gate.and(ctx, gt, le_prev_not);
            computed = gate.select(ctx, ys[j - 1], computed, cond_window);
        }

        // Apply boundary conditions
        let y_one = ctx.load_constant(fixed_point_chip.quantization(1.0));
        computed = gate.select(ctx, y_one, computed, ge_max);  // x >= max → 1.0
        // x < min → 0 (already default)

        // Compare with expected result
        let eq = gate.is_equal(ctx, computed, result_fp[i]);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
