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
    pub X: Vec<Vec<i64>>,
    pub result: Vec<Vec<f64>>,
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
    let range = builder.range_chip();
    let fixed = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    let _poseidon =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());

    let rows = input.X.len();
    let cols = input.X[0].len();

    // --- Load integer matrix X ---
    let mut X: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..rows {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..cols {
            let val = input.X[i][j];
            if val >= 0 {
                row.push(ctx.load_witness(F::from(val as u64)));
            } else {
                row.push(gate.neg(ctx, Constant(F::from((-val) as u64))));
            }
        }
        X.push(row);
    }

    // --- Load result matrix ---
    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..rows {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..cols {
            row.push(ctx.load_witness(fixed.quantization(input.result[i][j])));
        }
        result.push(row);
    }

    // --- Step 1: compute L1 norms per row ---
    let mut l1_vals: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..rows {
        let mut s = fixed.qadd(ctx, X[i][0], X[i][1]);
        for j in 2..cols {
            s = fixed.qadd(ctx, s, X[i][j]);
        }
        l1_vals.push(s);
    }

    // --- Step 2: normalize and assert correctness ---
    for i in 0..rows {
        for j in 0..cols {
            // expected = X[i][j] / l1[i]
            let numerator = X[i][j];
            let denominator = l1_vals[i];
            let expected = fixed.qdiv(ctx, numerator, denominator);
            let diff = fixed.qsub(ctx, expected, result[i][j]);
            let upper = range.is_less_than(ctx, diff, Constant(fixed.quantization(0.001)), 128);
            let lower = range.is_less_than(ctx, Constant(fixed.quantization(-0.001)), diff, 128);
            let eq = gate.and(ctx, upper, lower);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
