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
    pub x: f64,
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

    // --- Load inputs ---
    let x = ctx.load_witness(fixed_point_chip.quantization(input.x));
    let result = ctx.load_witness(fixed_point_chip.quantization(input.result));

    // --- Constants ---
    let x_min = Constant(fixed_point_chip.quantization(0.0));
    let x_max = Constant(fixed_point_chip.quantization(1.0));
    let three = Constant(fixed_point_chip.quantization(3.0));
    let two = Constant(fixed_point_chip.quantization(2.0));

    // --- Condition flags ---
    let gt_xmax = range_chip.is_less_than(ctx, x_max, x, 128); // x > x_max ?
    let lt_xmin = range_chip.is_less_than(ctx, x, x_min, 128); // x < x_min ?
    let ge_xmin = gate.not(ctx, lt_xmin);                      // x >= x_min

    // --- Compute 3*x^2 - 2*x^3 ---
    let x_sq = fixed_point_chip.qmul(ctx, x, x);
    let x_cu = fixed_point_chip.qmul(ctx, x_sq, x);
    let term1 = fixed_point_chip.qmul(ctx, three, x_sq);
    let term2 = fixed_point_chip.qmul(ctx, two, x_cu);
    let cubic_val = fixed_point_chip.qsub(ctx, term1, term2);

    // --- expected = x_min by default ---
    let mut expected = x_min;

    // If x >= x_min → expected = cubic_val
    expected = gate.select(ctx, cubic_val, expected, ge_xmin);
    // If x > x_max → expected = x_max
    expected = gate.select(ctx, x_max, expected, gt_xmax);

    // --- Compare with provided result (±1e-3 tolerance) ---
    let diff = fixed_point_chip.qsub(ctx, result, expected);
    let le = range_chip.is_less_than(ctx, diff, Constant(fixed_point_chip.quantization(0.001)), 128);
    let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
    let ok = gate.and(ctx, le, ge);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
