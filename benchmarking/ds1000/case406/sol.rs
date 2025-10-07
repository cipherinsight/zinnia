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
    pub index: Vec<u64>,
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

    // --- Step 1: Load arrays ---
    let n = input.a.len();
    assert_eq!(input.index.len(), n);

    let mut a_vals = Vec::new();
    let mut idx_vals = Vec::new();

    for i in 0..n {
        a_vals.push(ctx.load_witness(F::from(input.a[i])));
        idx_vals.push(ctx.load_witness(F::from(input.index[i])));
    }

    // --- Step 2: Initialize maxima for 3 groups (0,1,2) ---
    let mut max0 = ctx.load_constant(F::ZERO);
    let mut max1 = ctx.load_constant(F::ZERO);
    let mut max2 = ctx.load_constant(F::ZERO);

    // --- Step 3: Iterate through all entries ---
    for i in 0..n {
        let val = a_vals[i];
        let idx = idx_vals[i];

        // Conditions idx==0, idx==1, idx==2
        let is0 = gate.is_equal(ctx, idx, Constant(F::from(0)));
        let is1 = gate.is_equal(ctx, idx, Constant(F::from(1)));
        let is2 = gate.is_equal(ctx, idx, Constant(F::from(2)));

        // max update = select( cond, max(val, current), current )
        let gt0 = gate.is_bigger(ctx, val, max0, 128);
        let new0 = gate.select(ctx, val, max0, gt0);
        max0 = gate.select(ctx, new0, max0, is0);

        let gt1 = gate.is_bigger(ctx, val, max1, 128);
        let new1 = gate.select(ctx, val, max1, gt1);
        max1 = gate.select(ctx, new1, max1, is1);

        let gt2 = gate.is_bigger(ctx, val, max2, 128);
        let new2 = gate.select(ctx, val, max2, gt2);
        max2 = gate.select(ctx, new2, max2, is2);
    }

    // --- Step 4: Compare with provided result ---
    let exp0 = ctx.load_witness(F::from(input.result[0]));
    let exp1 = ctx.load_witness(F::from(input.result[1]));
    let exp2 = ctx.load_witness(F::from(input.result[2]));

    let eq0 = gate.is_equal(ctx, max0, exp0);
    let eq1 = gate.is_equal(ctx, max1, exp1);
    let eq2 = gate.is_equal(ctx, max2, exp2);

    gate.assert_is_const(ctx, &eq0, &F::ONE);
    gate.assert_is_const(ctx, &eq1, &F::ONE);
    gate.assert_is_const(ctx, &eq2, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
