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
    pub X: Vec<Vec<f64>>,        // shape: 5 x 4 (ints given, but loaded as fixed-point)
    pub result: Vec<Vec<f64>>,   // shape: 5 x 4
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

    let rows = 5usize;
    let cols = 4usize;

    // Load X (as fixed-point) and result
    let mut X: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(ctx.load_witness(fp.quantization(input.X[i][j])));
        }
        X.push(row);
    }

    let mut R: Vec<Vec<AssignedValue<F>>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(ctx.load_witness(fp.quantization(input.result[i][j])));
        }
        R.push(row);
    }

    // Tolerance constants
    let tol_pos = Constant(fp.quantization(0.001));
    let tol_neg = Constant(fp.quantization(-0.001));
    let zero = ctx.load_constant(fp.quantization(0.0));

    // 1) Compute per-row L1 norms: l1[i] = sum_j |X[i,j]|
    let mut l1: Vec<AssignedValue<F>> = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut s = zero;
        for j in 0..cols {
            let abs_ij = fp.qabs(ctx, X[i][j]);
            s = fp.qadd(ctx, s, abs_ij);
        }
        l1.push(s);
    }

    // 2) expected[i][j] = X[i][j] / l1[i]; compare to result within ±1e-3
    for i in 0..rows {
        for j in 0..cols {
            let num = X[i][j];
            let den = l1[i];
            let exp_ij = fp.qdiv(ctx, num, den);

            let diff = fp.qsub(ctx, R[i][j], exp_ij);
            let le = range.is_less_than(ctx, diff, tol_pos, 128);
            let ge = range.is_less_than(ctx, tol_neg, diff, 128);
            let ok = gate.and(ctx, le, ge);
            gate.assert_is_const(ctx, &ok, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
