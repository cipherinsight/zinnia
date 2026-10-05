use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
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
use halo2_base::poseidon::hasher::PoseidonHasher;
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub x: Vec<u64>,      // len = 9
    pub y: Vec<u64>,      // len = 9
    pub a: u64,
    pub b: u64,
    pub result: i64,      // could be -1 if not found
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63; // unused here; kept for parity with other examples
    let gate = GateChip::<F>::default();
    let _range = builder.range_chip();
    let _fp = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let n = 9usize;

    // Load arrays x, y
    let x: Vec<AssignedValue<F>> =
        input.x.iter().map(|v| ctx.load_witness(F::from(*v))).collect();
    let y: Vec<AssignedValue<F>> =
        input.y.iter().map(|v| ctx.load_witness(F::from(*v))).collect();

    // Load scalars a, b
    let a = ctx.load_witness(F::from(input.a));
    let b = ctx.load_witness(F::from(input.b));

    // Load expected result (may be negative)
    let out = if input.result >= 0 {
        ctx.load_witness(F::from(input.result as u64))
    } else {
        // load positive magnitude then negate
        let pos = ctx.load_witness(F::from((-input.result) as u64));
        gate.neg(ctx, pos)
    };

    // found_index = -1
    let neg_one = gate.neg(ctx, Constant(F::from(1u64)));
    let mut found = neg_one;

    // for i in 0..n:
    //   if x[i]==a && y[i]==b && found==-1: found = i
    for i in 0..n {
        let xi_eq_a = gate.is_equal(ctx, x[i], a);
        let yi_eq_b = gate.is_equal(ctx, y[i], b);
        let found_is_neg1 = gate.is_equal(ctx, found, neg_one);

        let t = gate.and(ctx, xi_eq_a, yi_eq_b);
        let cond = gate.and(ctx, t, found_is_neg1);

        // i as field
        let i_val = Constant(F::from(i as u64));

        // found = cond ? i : found
        found = gate.select(ctx, i_val, found, cond);
    }

    // assert result == found
    let eq = gate.is_equal(ctx, out, found);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
