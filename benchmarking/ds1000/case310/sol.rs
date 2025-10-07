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
    pub a: Vec<Vec<u64>>,
    pub result: u64,
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

    // Load matrix a
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    // Flatten a in C order: row-major
    let mut flat: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..a.len() {
        for j in 0..a[i].len() {
            flat.push(a[i][j]);
        }
    }

    // Compute argmin: track min_val and min_idx
    let mut min_val = flat[0];
    let mut min_idx = ctx.load_constant(F::from(0));

    for i in 1..flat.len() {
        let current = flat[i];
        let less = range_chip.is_less_than(ctx, current, min_val, 128);
        // Update value and index if smaller
        min_val = gate.select(ctx, current, min_val, less);
        let i_const = Constant(F::from(i as u64));
        min_idx = gate.select(ctx, i_const, min_idx, less);
    }

    // Compare computed min_idx with provided result
    let expected_result = ctx.load_witness(F::from(input.result));
    let eq = gate.is_equal(ctx, expected_result, min_idx);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
