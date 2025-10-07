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
    pub mask: Vec<Vec<u64>>,
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
    let range = builder.range_chip();
    let ctx = builder.main(0);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());

    // --- Load input matrices ---
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    let mut mask: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.mask.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.mask[i].len() {
            row.push(ctx.load_witness(F::from(input.mask[i][j])));
        }
        mask.push(row);
    }

    // --- Step 1: compute per-row maxima ---
    let mut row_max: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..3 {
        let gt = range.is_less_than(ctx, a[i][0], a[i][1], 128);
        // if a[i][0] < a[i][1], choose a[i][1], else a[i][0]
        let max_val = gate.select(ctx, a[i][1], a[i][0], gt);
        row_max.push(max_val);
    }

    // --- Step 2: check mask correctness ---
    // mask[i][j] == 1  <=>  a[i][j] == row_max[i]
    for i in 0..3 {
        for j in 0..2 {
            let eq_val = gate.is_equal(ctx, a[i][j], row_max[i]);
            let eq_mask = gate.is_equal(ctx, mask[i][j], Constant(F::ONE));
            let eq_bools = gate.is_equal(ctx, eq_val, eq_mask);
            gate.assert_is_const(ctx, &eq_bools, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
