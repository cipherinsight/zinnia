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
    pub number: u64,
    pub is_contained: u64,
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

    // --- Load inputs ---
    let a: Vec<AssignedValue<F>> =
        input.a.iter().map(|x| ctx.load_witness(F::from(*x))).collect();
    let number = ctx.load_witness(F::from(input.number));
    let expected = ctx.load_witness(F::from(input.is_contained));

    // --- Initialize found = 0 ---
    let mut found = ctx.load_constant(F::ZERO);

    // --- Step 1: Iterate and check containment ---
    for i in 0..a.len() {
        let eq = gate.is_equal(ctx, a[i], number);
        // found = found OR eq
        found = gate.or(ctx, found, eq);
    }

    // --- Step 2: Assert equality ---
    let eq_result = gate.is_equal(ctx, found, expected);
    gate.assert_is_const(ctx, &eq_result, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
