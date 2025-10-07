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
    pub data: Vec<u64>,
    pub result: Vec<u64>,
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

    // Load data
    let data: Vec<AssignedValue<F>> = input
        .data
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // Load result
    let result: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // --- Step 1: trim to multiple of bin_size ---
    let bin_size = 3;
    let trimmed_len = (data.len() / bin_size) * bin_size;
    let trimmed: Vec<AssignedValue<F>> = data.iter().take(trimmed_len).cloned().collect();

    // --- Step 2: reshape to (3, 3) and compute max per row ---
    let nrow = trimmed_len / bin_size;
    let mut bin_data_max: Vec<AssignedValue<F>> = Vec::new();

    for i in 0..nrow {
        // initial max = first element in this row
        let mut current_max = trimmed[i * bin_size];
        for j in 1..bin_size {
            let candidate = trimmed[i * bin_size + j];
            let less = range_chip.is_less_than(ctx, current_max, candidate, 128);
            // if current_max < candidate, update
            current_max = gate.select(ctx, candidate, current_max, less);
        }
        bin_data_max.push(current_max);
    }

    // --- Step 3: assert result == bin_data_max ---
    for i in 0..nrow {
        let eq = gate.is_equal(ctx, result[i], bin_data_max[i]);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
