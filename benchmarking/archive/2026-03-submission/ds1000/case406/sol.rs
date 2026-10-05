use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::Constant,
};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use serde::{Serialize, Deserialize};
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;
use halo2_base::poseidon::hasher::PoseidonHasher;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<u64>,        // len = 10
    pub index: Vec<u64>,    // len = 10, values in {0,1,2}
    pub result: Vec<u64>,   // len = 3
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63; // unused for ints, kept for parity
    let gate = GateChip::<F>::default();
    let range = builder.range_chip();
    let _fp = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load inputs
    let a: Vec<AssignedValue<F>> =
        input.a.iter().map(|x| ctx.load_witness(F::from(*x))).collect();
    let idxs: Vec<AssignedValue<F>> =
        input.index.iter().map(|x| ctx.load_witness(F::from(*x))).collect();
    let outs: Vec<AssignedValue<F>> =
        input.result.iter().map(|x| ctx.load_witness(F::from(*x))).collect();

    // max0 = max1 = max2 = 0
    let mut max0 = ctx.load_constant(F::from(0));
    let mut max1 = ctx.load_constant(F::from(0));
    let mut max2 = ctx.load_constant(F::from(0));

    // for i in 0..10:
    //   if idx[i]==g AND a[i] > maxg: maxg = a[i]
    for i in 0..10 {
        let ai = a[i];
        let ii = idxs[i];

        // group == 0,1,2 flags
        let is0 = gate.is_equal(ctx, ii, Constant(F::from(0)));
        let is1 = gate.is_equal(ctx, ii, Constant(F::from(1)));
        let is2 = gate.is_equal(ctx, ii, Constant(F::from(2)));

        // ai > maxg  <=>  maxg < ai
        let gt0 = range.is_less_than(ctx, max0, ai, 128);
        let gt1 = range.is_less_than(ctx, max1, ai, 128);
        let gt2 = range.is_less_than(ctx, max2, ai, 128);

        // both conditions must hold
        let c0 = gate.and(ctx, is0, gt0);
        let c1 = gate.and(ctx, is1, gt1);
        let c2 = gate.and(ctx, is2, gt2);

        // update maxima via select
        max0 = gate.select(ctx, ai, max0, c0);
        max1 = gate.select(ctx, ai, max1, c1);
        max2 = gate.select(ctx, ai, max2, c2);
    }

    // expected = [max0, max1, max2]; assert result == expected
    let eq0 = gate.is_equal(ctx, outs[0], max0);
    let eq1 = gate.is_equal(ctx, outs[1], max1);
    let eq2 = gate.is_equal(ctx, outs[2], max2);
    gate.assert_is_const(ctx, &eq0, &F::ONE);
    gate.assert_is_const(ctx, &eq1, &F::ONE);
    gate.assert_is_const(ctx, &eq2, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
