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
    pub x: Vec<u64>,
    pub y: Vec<u64>,
    pub a: u64,
    pub b: u64,
    pub result: i64,
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
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());

    // --- Load inputs ---
    let x: Vec<AssignedValue<F>> = input.x.iter().map(|v| ctx.load_witness(F::from(*v))).collect();
    let y: Vec<AssignedValue<F>> = input.y.iter().map(|v| ctx.load_witness(F::from(*v))).collect();
    let a = ctx.load_witness(F::from(input.a));
    let b = ctx.load_witness(F::from(input.b));
    let expected = ctx.load_witness(F::from(input.result));

    // Constants
    let neg_one = gate.neg(ctx, Constant(F::ONE)); // -1
    let zero = Constant(F::ZERO);

    // --- Initialize found_index = -1 ---
    let mut found_index = neg_one;

    // --- Iterate over elements ---
    for i in 0..x.len() {
        // cond1: x[i] == a
        let cond1 = gate.is_equal(ctx, x[i], a);
        // cond2: y[i] == b
        let cond2 = gate.is_equal(ctx, y[i], b);
        // cond3: found_index == -1
        let cond3 = gate.is_equal(ctx, found_index, neg_one);

        // combined condition: (x[i]==a) AND (y[i]==b) AND (found_index==-1)
        let cond12 = gate.and(ctx, cond1, cond2);
        let combined = gate.and(ctx, cond12, cond3);

        // Select: if combined then i else found_index
        let i_const = Constant(F::from(i as u64));
        found_index = gate.select(ctx, i_const, found_index, combined);
    }

    // --- Assert final result ---
    let eq = gate.is_equal(ctx, found_index, expected);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
