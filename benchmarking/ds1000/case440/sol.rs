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
    pub Y: Vec<Vec<Vec<f64>>>,
    pub X: Vec<Vec<f64>>,
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

    // --- Load inputs ---
    let N = input.Y.len(); // number of slices
    let M = input.X.len(); // number of rows

    let mut Y: Vec<Vec<Vec<AssignedValue<F>>>> = Vec::new();
    for i in 0..N {
        let mut slice: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for j in 0..M {
            let mut row: Vec<AssignedValue<F>> = Vec::new();
            for k in 0..M {
                row.push(ctx.load_witness(fixed.quantization(input.Y[i][j][k])));
            }
            slice.push(row);
        }
        Y.push(slice);
    }

    let mut X: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..M {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..N {
            row.push(ctx.load_witness(fixed.quantization(input.X[i][j])));
        }
        X.push(row);
    }

    // --- Step 1: verify that X[j,i]^2 == Y[i][j][j] ---
    for i in 0..N {
        for j in 0..M {
            let xji = X[j][i];
            let yjj = Y[i][j][j];
            let x_sq = fixed.qmul(ctx, xji, xji);
            let diff = fixed.qsub(ctx, x_sq, yjj);

            // |x^2 - Y[i][j][j]| < 1e-3  → equality within tolerance
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
