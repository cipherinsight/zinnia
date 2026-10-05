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
    pub a: Vec<u64>,
    pub result: Vec<Vec<u64>>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    let gate = GateChip::<F>::default();
    let ctx = builder.main(0);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());

    // --- Load inputs ---
    let a: Vec<AssignedValue<F>> =
        input.a.iter().map(|v| ctx.load_witness(F::from(*v))).collect();

    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for row in &input.result {
        result.push(row.iter().map(|v| ctx.load_witness(F::from(*v))).collect());
    }

    // --- Core verification ---
    for i in 0..a.len() {
        for j in 0..result[i].len() {
            // expected = 1 if a[i] == j else 0
            let eq = gate.is_equal(ctx, a[i], Constant(F::from(j as u64)));
            let eq_expected = gate.is_equal(ctx, result[i][j], Constant(F::from(1u64)));
            // Enforce result[i][j] == (a[i] == j)
            let check = gate.is_equal(ctx, eq, eq_expected);
            gate.assert_is_const(ctx, &check, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
