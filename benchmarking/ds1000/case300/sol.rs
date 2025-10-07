use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Existing, Witness},
};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
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
    pub a: Vec<f64>,
    pub p: f64,
    pub result: f64,
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
    let mut poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load inputs
    let a: Vec<AssignedValue<F>> = input
        .a
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();
    let p = ctx.load_witness(fixed_point_chip.quantization(input.p));
    let result = ctx.load_witness(fixed_point_chip.quantization(input.result));

    let n = 5;
    let one = Constant(fixed_point_chip.quantization(1.0));
    let hundred = Constant(fixed_point_chip.quantization(100.0));

    // Verify non-decreasing order: a[i] <= a[i+1]
    for i in 0..(n - 1) {
        let cond = range_chip.is_less_than(ctx, a[i], a[i + 1], 128);
        gate.assert_is_const(ctx, &cond, &F::ONE);
    }

    // rank = (p / 100.0) * (n - 1)
    let div = fixed_point_chip.qdiv(ctx, p, hundred);
    let n_minus_1 = Constant(fixed_point_chip.quantization((n - 1) as f64));
    let rank = fixed_point_chip.qmul(ctx, div, n_minus_1);

    // lower = int(rank)
    // upper = lower + 1
    // fraction = rank - lower
    // (Note: fixed-point int cast → floor)
    let lower = fixed_point_chip.qfloor(ctx, rank);
    let upper = fixed_point_chip.qadd(ctx, lower, Constant(fixed_point_chip.quantization(1.0)));
    let fraction = fixed_point_chip.qsub(ctx, rank, lower);

    // a[lower], a[upper]
    // We approximate array access with select loops
    let mut a_lower = ctx.load_constant(fixed_point_chip.quantization(0.0));
    let mut a_upper = ctx.load_constant(fixed_point_chip.quantization(0.0));

    for i in 0..n {
        let idx_const = Constant(fixed_point_chip.quantization(i as f64));
        let eq_lower = fixed_point_chip.qeq(ctx, lower, idx_const);
        let eq_upper = fixed_point_chip.qeq(ctx, upper, idx_const);
        a_lower = fixed_point_chip.qselect(ctx, a[i], a_lower, eq_lower);
        a_upper = fixed_point_chip.qselect(ctx, a[i], a_upper, eq_upper);
    }

    // interpolated = a_lower + (a_upper - a_lower) * fraction
    let diff = fixed_point_chip.qsub(ctx, a_upper, a_lower);
    let scaled = fixed_point_chip.qmul(ctx, diff, fraction);
    let interpolated = fixed_point_chip.qadd(ctx, a_lower, scaled);

    // assert result == interpolated (within ±1e-3)
    let delta = fixed_point_chip.qsub(ctx, result, interpolated);
    let le = range_chip.is_less_than(ctx, delta, Constant(fixed_point_chip.quantization(0.001)), 128);
    let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), delta, 128);
    let ok = gate.and(ctx, le, ge);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
