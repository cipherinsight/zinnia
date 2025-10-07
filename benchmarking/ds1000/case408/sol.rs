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
    let range_chip = builder.range_chip();
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let n = input.a.len();
    assert_eq!(input.index.len(), n);

    // --- Load inputs ---
    let mut a_vals = Vec::new();
    let mut idx_vals = Vec::new();
    for i in 0..n {
        a_vals.push(ctx.load_witness(F::from(input.a[i])));
        idx_vals.push(ctx.load_witness(F::from(input.index[i])));
    }

    // --- Initialize found flags and minima ---
    let mut found0 = ctx.load_constant(F::ZERO);
    let mut found1 = ctx.load_constant(F::ZERO);
    let mut found2 = ctx.load_constant(F::ZERO);

    let mut min0 = ctx.load_constant(F::ZERO);
    let mut min1 = ctx.load_constant(F::ZERO);
    let mut min2 = ctx.load_constant(F::ZERO);

    // --- First pass: seed minima from first occurrence ---
    for i in 0..n {
        let idx = idx_vals[i];
        let val = a_vals[i];

        let is0 = gate.is_equal(ctx, idx, Constant(F::from(0)));
        let is1 = gate.is_equal(ctx, idx, Constant(F::from(1)));
        let is2 = gate.is_equal(ctx, idx, Constant(F::from(2)));

        let notf0 = gate.not(ctx, found0);
        let notf1 = gate.not(ctx, found1);
        let notf2 = gate.not(ctx, found2);

        let cond0 = gate.and(ctx, is0, notf0);
        let cond1 = gate.and(ctx, is1, notf1);
        let cond2 = gate.and(ctx, is2, notf2);

        min0 = gate.select(ctx, val, min0, cond0);
        min1 = gate.select(ctx, val, min1, cond1);
        min2 = gate.select(ctx, val, min2, cond2);

        found0 = gate.or(ctx, found0, is0);
        found1 = gate.or(ctx, found1, is1);
        found2 = gate.or(ctx, found2, is2);
    }

    // --- Require that each group exists ---
    gate.assert_is_const(ctx, &found0, &F::ONE);
    gate.assert_is_const(ctx, &found1, &F::ONE);
    gate.assert_is_const(ctx, &found2, &F::ONE);

    // --- Second pass: refine minima ---
    for i in 0..n {
        let idx = idx_vals[i];
        let val = a_vals[i];

        let is0 = gate.is_equal(ctx, idx, Constant(F::from(0)));
        let is1 = gate.is_equal(ctx, idx, Constant(F::from(1)));
        let is2 = gate.is_equal(ctx, idx, Constant(F::from(2)));

        // val < min? → range_chip.is_less_than(ctx, val, min, 128)
        let lt0 = range_chip.is_less_than(ctx, val, min0, 128);
        let lt1 = range_chip.is_less_than(ctx, val, min1, 128);
        let lt2 = range_chip.is_less_than(ctx, val, min2, 128);

        let cond0 = gate.and(ctx, is0, lt0);
        let cond1 = gate.and(ctx, is1, lt1);
        let cond2 = gate.and(ctx, is2, lt2);

        min0 = gate.select(ctx, val, min0, cond0);
        min1 = gate.select(ctx, val, min1, cond1);
        min2 = gate.select(ctx, val, min2, cond2);
    }

    // --- Compare results ---
    let exp0 = ctx.load_witness(F::from(input.result[0]));
    let exp1 = ctx.load_witness(F::from(input.result[1]));
    let exp2 = ctx.load_witness(F::from(input.result[2]));

    let eq0 = gate.is_equal(ctx, min0, exp0);
    let eq1 = gate.is_equal(ctx, min1, exp1);
    let eq2 = gate.is_equal(ctx, min2, exp2);

    gate.assert_is_const(ctx, &eq0, &F::ONE);
    gate.assert_is_const(ctx, &eq1, &F::ONE);
    gate.assert_is_const(ctx, &eq2, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
