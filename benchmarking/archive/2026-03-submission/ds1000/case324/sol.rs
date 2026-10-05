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
    pub degree: f64,
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

    // --- Load inputs ---
    let degree = ctx.load_witness(fixed_point_chip.quantization(input.degree));
    let result = ctx.load_witness(fixed_point_chip.quantization(input.result));

    // --- Constants ---
    let pi = Constant(fixed_point_chip.quantization(3.141592653589793));
    let const_180 = Constant(fixed_point_chip.quantization(180.0));
    let tol = Constant(fixed_point_chip.quantization(0.001));

    // --- rad = degree * π / 180 ---
    let tmp = fixed_point_chip.qmul(ctx, degree, pi);
    let rad = fixed_point_chip.qdiv(ctx, tmp, const_180);

    // --- computed = cos(rad) ---
    let computed = fixed_point_chip.qcos(ctx, rad);

    // --- assert result == computed (within ±0.001) ---
    let diff = fixed_point_chip.qsub(ctx, result, computed);
    let le = range_chip.is_less_than(ctx, diff, tol, 128);
    let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
    let ok = gate.and(ctx, le, ge);
    gate.assert_is_const(ctx, &ok, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
