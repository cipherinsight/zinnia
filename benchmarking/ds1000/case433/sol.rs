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
    pub result: Vec<Vec<u64>>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    let gate = GateChip::<F>::default();

    // Borrow from `builder` BEFORE getting `ctx`
    let _range = builder.range_chip();
    let _poseidon = PoseidonHasher::<F, T, RATE>::new(
        OptimizedPoseidonSpec::new::<R_F, R_P, 0>(),
    );

    // Now take the mutable borrow for the context
    let ctx = builder.main(0);

    // --- Load matrices ---
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.result.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.result[i].len() {
            row.push(ctx.load_witness(F::from(input.result[i][j])));
        }
        result.push(row);
    }

    // --- Parameters ---
    let zero_rows = 0u64;
    let zero_cols = 0u64;
    let zero_const = Constant(F::ZERO);

    // --- Step 1 & 2: build modified matrix and assert equals result ---
    // modified[i][j] = 0 if (i == zero_rows) or (j == zero_cols) else a[i][j]
    for i in 0..4 {
        for j in 0..4 {
            let i_eq = gate.is_equal(ctx, Constant(F::from(i as u64)), Constant(F::from(zero_rows)));
            let j_eq = gate.is_equal(ctx, Constant(F::from(j as u64)), Constant(F::from(zero_cols)));
            let is_zero_pos = gate.or(ctx, i_eq, j_eq);
            let selected = gate.select(ctx, zero_const, a[i][j], is_zero_pos);

            let eq = gate.is_equal(ctx, selected, result[i][j]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
