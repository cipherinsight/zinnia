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
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub c: Vec<u64>,
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
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load inputs
    let a: Vec<AssignedValue<F>> = input
        .a
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();
    let b: Vec<AssignedValue<F>> = input
        .b
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();
    let c: Vec<AssignedValue<F>> = input
        .c
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();
    let result: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    let n = a.len();

    // Element-wise max of [a,b,c]
    for i in 0..n {
        // Step 1: tmp = max(a[i], b[i])
        let less_ab = range_chip.is_less_than(ctx, a[i], b[i], 128);
        let tmp = gate.select(ctx, b[i], a[i], less_ab);

        // Step 2: computed = max(tmp, c[i])
        let less_tmpc = range_chip.is_less_than(ctx, tmp, c[i], 128);
        let computed = gate.select(ctx, c[i], tmp, less_tmpc);

        // Step 3: assert equality with expected result
        let eq = gate.is_equal(ctx, result[i], computed);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
