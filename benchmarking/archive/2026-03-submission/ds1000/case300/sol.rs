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
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Load array a ---
    let a: Vec<AssignedValue<F>> = input
        .a
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();

    let n = a.len();
    let p = ctx.load_witness(fixed_point_chip.quantization(input.p));
    let result = ctx.load_witness(fixed_point_chip.quantization(input.result));

    // --- Step 1: ensure sortedness ---
    for i in 0..(n - 1) {
        let less = range_chip.is_less_than(ctx, a[i + 1], a[i], 128);
        let not_less = gate.not(ctx, less);
        gate.assert_is_const(ctx, &not_less, &F::ONE);
    }

    // --- Step 2: rank = (p / 100) * (n - 1) ---
    let const_100 = Constant(fixed_point_chip.quantization(100.0));
    let n_minus_1 = Constant(fixed_point_chip.quantization((n - 1) as f64));
    let p_div_100 = fixed_point_chip.qdiv(ctx, p, const_100);
    let rank = fixed_point_chip.qmul(ctx, p_div_100, n_minus_1);

    // --- Step 3: In-circuit floor(rank) ---
    let mut lower_idx = ctx.load_constant(F::from(0));
    for i in 0..(n - 1) {
        let i_const = Constant(fixed_point_chip.quantization(i as f64));
        let ge = range_chip.is_less_than(ctx, i_const, rank, 128); // rank >= i ?
        lower_idx = gate.select(ctx, Constant(F::from(i as u64)), lower_idx, ge);
    }

    let upper_idx = gate.add(ctx, lower_idx, Constant(F::from(1u64)));

    // --- Step 4: fraction = rank - lower_idx ---
    let lower_val_f64 = fixed_point_chip.dequantization(*lower_idx.value());
    let lower_quant = fixed_point_chip.quantization(lower_val_f64);
    let lower_q = ctx.load_witness(lower_quant);
    let fraction = fixed_point_chip.qsub(ctx, rank, lower_q);

    // --- Step 5: interpolate (fixed borrow version) ---
    let mut a_lower = ctx.load_constant(F::from(0));
    let mut a_upper = ctx.load_constant(F::from(0));
    for t in 0..n {
        let t_const = Constant(F::from(t as u64));
        let eq_lower = gate.is_equal(ctx, lower_idx, t_const);
        let eq_upper = gate.is_equal(ctx, upper_idx, t_const);

        // inner ops separated to avoid borrow overlap
        let mul_lower = gate.mul(ctx, a[t], eq_lower);
        a_lower = gate.add(ctx, a_lower, mul_lower);

        let mul_upper = gate.mul(ctx, a[t], eq_upper);
        a_upper = gate.add(ctx, a_upper, mul_upper);
    }

    let diff = fixed_point_chip.qsub(ctx, a_upper, a_lower);
    let scaled = fixed_point_chip.qmul(ctx, diff, fraction);
    let interpolated = fixed_point_chip.qadd(ctx, a_lower, scaled);

    // --- Step 6: assert |result - interpolated| <= 1e-3 ---
    let diff_res = fixed_point_chip.qsub(ctx, result, interpolated);
    let tol = Constant(fixed_point_chip.quantization(0.001));
    let upper = range_chip.is_less_than(ctx, diff_res, tol, 128);
    let lower = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff_res, 128);
    let ok = gate.and(ctx, upper, lower);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
