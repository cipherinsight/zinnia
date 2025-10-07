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
    pub accmap: Vec<u64>,
    pub result: Vec<u64>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Step 1: Load inputs ---
    let n = input.a.len();
    assert_eq!(input.accmap.len(), n);

    let mut a_vals = Vec::new();
    let mut accmap_vals = Vec::new();

    for i in 0..n {
        a_vals.push(ctx.load_witness(F::from(input.a[i])));
        accmap_vals.push(ctx.load_witness(F::from(input.accmap[i])));
    }

    // --- Step 2: Initialize sums for groups 0,1,2 ---
    let mut sum0 = ctx.load_constant(F::ZERO);
    let mut sum1 = ctx.load_constant(F::ZERO);
    let mut sum2 = ctx.load_constant(F::ZERO);

    // --- Step 3: Aggregate according to accmap ---
    for i in 0..n {
        let val = a_vals[i];
        let idx = accmap_vals[i];

        let is0 = gate.is_equal(ctx, idx, Constant(F::from(0)));
        let is1 = gate.is_equal(ctx, idx, Constant(F::from(1)));
        let is2 = gate.is_equal(ctx, idx, Constant(F::from(2)));

        // add if condition true: sum = sum + val * cond
        let add0 = gate.mul(ctx, val, is0);
        let add1 = gate.mul(ctx, val, is1);
        let add2 = gate.mul(ctx, val, is2);

        sum0 = gate.add(ctx, sum0, add0);
        sum1 = gate.add(ctx, sum1, add1);
        sum2 = gate.add(ctx, sum2, add2);
    }

    // --- Step 4: Compare with expected results ---
    let exp0 = ctx.load_witness(F::from(input.result[0]));
    let exp1 = ctx.load_witness(F::from(input.result[1]));
    let exp2 = ctx.load_witness(F::from(input.result[2]));

    let eq0 = gate.is_equal(ctx, sum0, exp0);
    let eq1 = gate.is_equal(ctx, sum1, exp1);
    let eq2 = gate.is_equal(ctx, sum2, exp2);

    gate.assert_is_const(ctx, &eq0, &F::ONE);
    gate.assert_is_const(ctx, &eq1, &F::ONE);
    gate.assert_is_const(ctx, &eq2, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
