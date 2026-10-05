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
    pub grades: Vec<f64>,
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
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Step 1: Load inputs ---
    let n = input.grades.len();
    let grades: Vec<AssignedValue<F>> = input
        .grades
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();
    let results: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();

    // --- Step 2: Validate sortedness (non-decreasing) ---
    for i in 0..(n - 1) {
        let leq = range_chip.is_less_than(ctx, grades[i + 1], grades[i], 128);
        let not_leq = gate.not(ctx, leq);
        gate.assert_is_const(ctx, &not_leq, &F::ONE);
    }

    // --- Step 3: Verify ECDF values result[i] = (i+1)/n ---
    let n_const = Constant(fixed_point_chip.quantization(n as f64));
    for i in 0..n {
        let idx_plus = (i + 1) as f64;
        let idx_val = Constant(fixed_point_chip.quantization(idx_plus));
        let expected = fixed_point_chip.qdiv(ctx, idx_val, n_const);
        let diff = fixed_point_chip.qsub(ctx, results[i], expected);

        // assert |diff| <= 1e-3
        let upper = range_chip.is_less_than(ctx, diff, Constant(fixed_point_chip.quantization(0.001)), 128);
        let lower = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
        let within_tol = gate.and(ctx, upper, lower);
        gate.assert_is_const(ctx, &within_tol, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
