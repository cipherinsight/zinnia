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
    let _poseidon = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());

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

    // --- Zeroing parameters ---
    let zero_row = 1u64;
    let zero_col = 0u64;
    let zero_const = Constant(F::ZERO);

    // --- Step 1: Construct modified matrix ---
    for i in 0..4 {
        for j in 0..4 {
            // row_match = (i == zero_row)
            let row_match = gate.is_equal(ctx, Constant(F::from(i as u64)), Constant(F::from(zero_row)));
            // col_match = (j == zero_col)
            let col_match = gate.is_equal(ctx, Constant(F::from(j as u64)), Constant(F::from(zero_col)));
            // should_zero = row_match OR col_match
            let should_zero = gate.or(ctx, row_match, col_match);

            // modified[i][j] = SELECT(0, a[i][j], should_zero)
            let modified = gate.select(ctx, zero_const, a[i][j], should_zero);

            // --- Step 2: assert modified == result[i][j] ---
            let eq = gate.is_equal(ctx, modified, result[i][j]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
