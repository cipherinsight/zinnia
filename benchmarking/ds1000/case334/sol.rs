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
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub c: Vec<u64>,
    pub result: Vec<f64>,
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
    let a: Vec<AssignedValue<F>> =
        input.a.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x as f64))).collect();
    let b: Vec<AssignedValue<F>> =
        input.b.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x as f64))).collect();
    let c: Vec<AssignedValue<F>> =
        input.c.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x as f64))).collect();
    let result: Vec<AssignedValue<F>> =
        input.result.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect();

    // --- Compute elementwise mean across a,b,c ---
    // result[i] == (a[i] + b[i] + c[i]) / 3
    let three = Constant(fixed_point_chip.quantization(3.0));

    for i in 0..a.len() {
        let mut sum = fixed_point_chip.qadd(ctx, a[i], b[i]);
        sum = fixed_point_chip.qadd(ctx, sum, c[i]);
        let mean = fixed_point_chip.qdiv(ctx, sum, three);

        // --- Verify equality within ±1e-3 ---
        let diff = fixed_point_chip.qsub(ctx, result[i], mean);
        let le = range_chip.is_less_than(ctx, diff, Constant(fixed_point_chip.quantization(0.001)), 128);
        let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
        let eq = gate.and(ctx, le, ge);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
