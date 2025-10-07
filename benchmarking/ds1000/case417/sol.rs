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
    pub data: Vec<f64>,
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
    let mut poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Load data & result ---
    let data: Vec<AssignedValue<F>> = input
        .data
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();
    let result: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();

    let bin_size = 3;
    let n = data.len();
    let trim_len = (n / bin_size) * bin_size; // 9

    // --- Step 1: Reverse ---
    let mut reversed: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..n {
        reversed.push(data[n - 1 - i]);
    }

    // --- Step 2: Trim ---
    let trimmed: Vec<AssignedValue<F>> = reversed.iter().take(trim_len).cloned().collect();

    // --- Step 3: Reshape (3, 3) & compute mean along last axis ---
    let nrow = trim_len / bin_size;
    let three_const = Constant(fixed_point_chip.quantization(3.0));

    assert!(nrow == 3);
    for i in 0..nrow {
        // Compute mean of 3 elements per row
        let mut sum = fixed_point_chip.qadd(ctx, trimmed[i * bin_size], trimmed[i * bin_size + 1]);
        sum = fixed_point_chip.qadd(ctx, sum, trimmed[i * bin_size + 2]);
        let mean = fixed_point_chip.qdiv(ctx, sum, three_const);

        // --- Compare with expected result[i] (within ±1e-3) ---
        let diff = fixed_point_chip.qsub(ctx, result[i], mean);
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
