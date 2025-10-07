use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Existing, Witness},
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
    pub data: Vec<Vec<f64>>,
    pub result: Vec<Vec<f64>>,
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
    let mut poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Load inputs ---
    let mut data: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.data.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.data[i].len() {
            row.push(ctx.load_witness(fixed_point_chip.quantization(input.data[i][j])));
        }
        data.push(row);
    }

    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.result.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.result[i].len() {
            row.push(ctx.load_witness(fixed_point_chip.quantization(input.result[i][j])));
        }
        result.push(row);
    }

    // --- Parameters ---
    let bin_size = 3;
    let one_third = Constant(fixed_point_chip.quantization(1.0 / 3.0));

    // --- Step 1: Reverse each row ---
    let mut reversed_rows: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..data.len() {
        let mut reversed_row: Vec<AssignedValue<F>> = Vec::new();
        let ncol = data[i].len();
        for j in 0..ncol {
            reversed_row.push(data[i][ncol - 1 - j]);
        }
        reversed_rows.push(reversed_row);
    }

    // --- Step 2: Trim to multiple of bin_size along columns ---
    let trim_len = (5 / bin_size) * bin_size; // 3
    let mut trimmed_rows: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..reversed_rows.len() {
        let row = reversed_rows[i]
            .iter()
            .take(trim_len)
            .cloned()
            .collect::<Vec<_>>();
        trimmed_rows.push(row);
    }

    // --- Step 3: Compute mean across each 3-element bin (axis=-1) ---
    let mut means: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..trimmed_rows.len() {
        let sum1 = fixed_point_chip.qadd(ctx, trimmed_rows[i][0], trimmed_rows[i][1]);
        let sum2 = fixed_point_chip.qadd(ctx, sum1, trimmed_rows[i][2]);
        let mean = fixed_point_chip.qmul(ctx, sum2, one_third);
        means.push(mean);
    }

    // --- Step 4: Reverse result array (same shape [[6],[5]]) ---
    let mut reversed_means: Vec<AssignedValue<F>> = means.clone();
    reversed_means.reverse();

    // --- Step 5: Assert equality ---
    for i in 0..reversed_means.len() {
        let diff = fixed_point_chip.qsub(ctx, result[i][0], reversed_means[i]);
        let le = range_chip.is_less_than(ctx, diff, Constant(fixed_point_chip.quantization(0.001)), 128);
        let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
        let eq = gate.and(ctx, le, ge);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
