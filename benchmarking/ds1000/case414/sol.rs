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
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;
use halo2_base::poseidon::hasher::PoseidonHasher;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub data: Vec<f64>,     // len = 10
    pub result: Vec<f64>,   // len = 3
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
    let data: Vec<AssignedValue<F>> = input
        .data
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();

    let results: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(fp.quantization(*x)))
        .collect();

    // Constants
    let bin_size = 3usize;
    let rows = 3usize; // (10 // 3) = 3
    let cols = bin_size;
    let zero = ctx.load_constant(fp.quantization(0.0));
    let three = Constant(fp.quantization(3.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));

    // Compute row means of reshaped trimmed data: reshape((3,3)) then mean along axis=1
    let mut means: Vec<AssignedValue<F>> = Vec::with_capacity(rows);
    for r in 0..rows {
        // sum row r over columns c=0..2; original index idx = r*cols + c
        let mut sum_r = zero;
        for c in 0..cols {
            let idx = r * cols + c; // uses only first 9 elements; data[9] is dropped
            sum_r = fp.qadd(ctx, sum_r, data[idx]);
        }
        let mean_r = fp.qdiv(ctx, sum_r, three);
        means.push(mean_r);
    }

    // Compare each result[i] with means[i] within ±1e-3
    for i in 0..rows {
        let diff = fp.qsub(ctx, results[i], means[i]);
        let le = range.is_less_than(ctx, diff, tol_pos, 128);
        let ge = range.is_less_than(ctx, tol_neg, diff, 128);
        let ok = gate.and(ctx, le, ge);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
