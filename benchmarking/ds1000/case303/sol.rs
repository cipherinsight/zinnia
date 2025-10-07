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
use halo2_graph::gadget::fixed_point::FixedPointChip;
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
    pub A: Vec<u64>,       // [1, 2, 3, 4, 5, 6, 7]
    pub B: Vec<Vec<u64>>,  // [[1,2],[3,4],[5,6]]
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
    let _range_chip = builder.range_chip();
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Load A ---
    let A: Vec<AssignedValue<F>> = input
        .A
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // --- Load B ---
    let mut B: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.B.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.B[i].len() {
            row.push(ctx.load_witness(F::from(input.B[i][j])));
        }
        B.push(row);
    }

    // --- Constants ---
    let ncol = 2;
    let nrow = 3;

    // --- Step 1: Truncate last element inside circuit ---
    // truncated = A[0..6]
    let mut truncated: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..(A.len() - 1) {
        truncated.push(A[i]);
    }

    // --- Step 2: Verify reshape correctness ---
    for i in 0..nrow {
        for j in 0..ncol {
            // idx = i * ncol + j
            let i_const = Constant(F::from(i as u64));
            let j_const = Constant(F::from(j as u64));
            let ncol_const = Constant(F::from(ncol as u64));
            let prod = gate.mul(ctx, i_const, ncol_const);
            let idx = gate.add(ctx, prod, j_const);

            // One-hot selection for truncated[idx]
            let mut selected = ctx.load_constant(F::from(0));
            for k in 0..truncated.len() {
                let k_const = Constant(F::from(k as u64));
                let eq = gate.is_equal(ctx, idx, k_const);
                let val = gate.mul(ctx, truncated[k], eq);
                selected = gate.add(ctx, selected, val);
            }

            // Enforce equality: B[i][j] == truncated[idx]
            let eq_check = gate.is_equal(ctx, B[i][j], selected);
            gate.assert_is_const(ctx, &eq_check, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
