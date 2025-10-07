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
    pub result: (f64, f64),
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

    // --- Load array and results ---
    let a: Vec<AssignedValue<F>> =
        input.a.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect();
    let result_lower = ctx.load_witness(fixed_point_chip.quantization(input.result.0));
    let result_upper = ctx.load_witness(fixed_point_chip.quantization(input.result.1));

    let n = a.len() as f64;
    let n_const = Constant(fixed_point_chip.quantization(n));
    let three = Constant(fixed_point_chip.quantization(3.0));

    // --- Step 1: mean_val = sum(a) / n ---
    let mut sum = fixed_point_chip.qadd(ctx, a[0], a[1]);
    for i in 2..a.len() {
        sum = fixed_point_chip.qadd(ctx, sum, a[i]);
    }
    let mean_val = fixed_point_chip.qdiv(ctx, sum, n_const);

    // --- Step 2: variance = sum((x - mean)^2) / n ---
    let mut var_sum = ctx.load_constant(fixed_point_chip.quantization(0.0));
    for i in 0..a.len() {
        let diff = fixed_point_chip.qsub(ctx, a[i], mean_val);
        let sq = fixed_point_chip.qmul(ctx, diff, diff);
        var_sum = fixed_point_chip.qadd(ctx, var_sum, sq);
    }
    let variance = fixed_point_chip.qdiv(ctx, var_sum, n_const);

    // --- Step 3: std_val = sqrt(variance) ---
    let std_val = fixed_point_chip.qsqrt(ctx, variance);

    // --- Step 4: lower = mean - 3 * std ---
    let three_std = fixed_point_chip.qmul(ctx, three, std_val);
    let lower = fixed_point_chip.qsub(ctx, mean_val, three_std);

    // --- Step 5: upper = mean + 3 * std ---
    let upper = fixed_point_chip.qadd(ctx, mean_val, three_std);

    // --- Step 6: assert equality within ±0.001 ---
    for (computed, expected) in [(lower, result_lower), (upper, result_upper)] {
        let diff = fixed_point_chip.qsub(ctx, expected, computed);
        let le = range_chip.is_less_than(ctx, diff, Constant(fixed_point_chip.quantization(0.001)), 128);
        let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
        let ok = gate.and(ctx, le, ge);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
