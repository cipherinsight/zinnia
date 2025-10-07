use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Witness},
};
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

    // --- Zero index sets (constants) ---
    let zero_rows = [1u64, 3u64];
    let zero_cols = [1u64, 2u64];
    let zero_const = Constant(F::ZERO);

    // --- Step 1: Construct modified matrix ---
    // For each cell (i, j):
    //   row_match = OR(i == r for r in zero_rows)
    //   col_match = OR(j == c for c in zero_cols)
    //   should_zero = row_match OR col_match
    //   modified[i][j] = SELECT(0, a[i][j], should_zero)
    for i in 0..4 {
        for j in 0..4 {
            // Check row membership
            let mut row_match = gate.is_equal(ctx, Constant(F::from(i as u64)), Constant(F::from(zero_rows[0])));
            for k in 1..zero_rows.len() {
                let eq = gate.is_equal(ctx, Constant(F::from(i as u64)), Constant(F::from(zero_rows[k])));
                row_match = gate.or(ctx, row_match, eq);
            }

            // Check column membership
            let mut col_match = gate.is_equal(ctx, Constant(F::from(j as u64)), Constant(F::from(zero_cols[0])));
            for k in 1..zero_cols.len() {
                let eq = gate.is_equal(ctx, Constant(F::from(j as u64)), Constant(F::from(zero_cols[k])));
                col_match = gate.or(ctx, col_match, eq);
            }

            // Combine conditions
            let should_zero = gate.or(ctx, row_match, col_match);

            // Select value
            let modified_val = gate.select(ctx, zero_const, a[i][j], should_zero);

            // --- Step 2: Assert equality with result[i][j] ---
            let eq = gate.is_equal(ctx, modified_val, result[i][j]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
