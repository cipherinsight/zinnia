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
    pub data: Vec<Vec<f64>>,   // shape: 2 x 5
    pub result: Vec<Vec<f64>>, // shape: 2 x 1
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
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load inputs
    let mut data: Vec<Vec<AssignedValue<F>>> = Vec::new();   // 2 x 5
    for i in 0..input.data.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.data[i].len() {
            row.push(ctx.load_witness(fp.quantization(input.data[i][j])));
        }
        data.push(row);
    }

    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new(); // 2 x 1
    for i in 0..input.result.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.result[i].len() {
            row.push(ctx.load_witness(fp.quantization(input.result[i][j])));
        }
        result.push(row);
    }

    // Constants
    let three = Constant(fp.quantization(3.0));
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));
    let zero = ctx.load_constant(fp.quantization(0.0));

    // rows = 2; bin_size = 3; trimmed = data[:, :3]; reshape -> (rows,1,3); mean over last axis
    let rows = 2usize;
    for i in 0..rows {
        // sum of first 3 entries in row i (indices 0,1,2)
        let mut sum = zero;
        let t0 = data[i][0];
        sum = fp.qadd(ctx, sum, t0);
        let t1 = data[i][1];
        sum = fp.qadd(ctx, sum, t1);
        let t2 = data[i][2];
        sum = fp.qadd(ctx, sum, t2);

        // mean = sum / 3
        let mean = fp.qdiv(ctx, sum, three);

        // compare to result[i][0] within ±1e-3
        let diff = fp.qsub(ctx, result[i][0], mean);
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
