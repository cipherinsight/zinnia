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
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
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
    pub x: f64,
    pub result: f64,
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
    let range = builder.range_chip();
    let fp = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load inputs
    let x = ctx.load_witness(fp.quantization(input.x));
    let out = ctx.load_witness(fp.quantization(input.result));

    // Constants
    let x_min = ctx.load_constant(fp.quantization(0.0));
    let x_max = ctx.load_constant(fp.quantization(1.0));
    let c2 = Constant(fp.quantization(2.0));
    let c3 = Constant(fp.quantization(3.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));

    // expected = x_min
    let mut expected = x_min;

    // cond1: x > x_max  <=>  x_max < x
    let cond_gt_xmax = range.is_less_than(ctx, x_max, x, 128);
    expected = gate.select(ctx, x_max, expected, cond_gt_xmax);

    // cond2: x >= x_min  <=>  NOT(x < x_min)
    let lt_xmin = range.is_less_than(ctx, x, x_min, 128);
    let ge_xmin = gate.not(ctx, lt_xmin);

    // Only apply cubic if NOT(cond1) AND ge_xmin
    let not_cond1 = gate.not(ctx, cond_gt_xmax);
    let cond_mid = gate.and(ctx, not_cond1, ge_xmin);

    // smooth = 3*x^2 - 2*x^3
    let x2 = fp.qmul(ctx, x, x);
    let x3 = fp.qmul(ctx, x2, x);
    let t1 = fp.qmul(ctx, c3, x2);
    let t2 = fp.qmul(ctx, c2, x3);
    let smooth = fp.qsub(ctx, t1, t2);

    expected = gate.select(ctx, smooth, expected, cond_mid);

    // Assert result == expected (±1e-3)
    let diff = fp.qsub(ctx, out, expected);
    let le = range.is_less_than(ctx, diff, tol_pos, 128);
    let ge = range.is_less_than(ctx, tol_neg, diff, 128);
    let ok = gate.and(ctx, le, ge);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
