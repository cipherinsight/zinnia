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
    pub A: Vec<u64>,
    pub B: Vec<Vec<u64>>,
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

    // Load A (flattened 1D)
    let A: Vec<AssignedValue<F>> = input
        .A
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // Load B (2D)
    let mut B: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.B.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.B[i].len() {
            row.push(ctx.load_witness(F::from(input.B[i][j])));
        }
        B.push(row);
    }

    let nrow = 3;
    let ncol = 2;

    // Verify reshape correctness:
    // for i in range(nrow):
    //     for j in range(ncol):
    //         idx = i * ncol + j
    //         assert B[i][j] == A[idx]
    for i in 0..nrow {
        for j in 0..ncol {
            let i_const = Constant(F::from(i as u64));
            let j_const = Constant(F::from(j as u64));
            let ncol_const = Constant(F::from(ncol as u64));
            let i_mul_ncol = gate.mul(ctx, i_const, ncol_const);
            let idx = gate.add(ctx, i_mul_ncol, j_const);

            // Select A[idx] by looping and matching index
            let mut A_idx = ctx.load_constant(F::from(0));
            for k in 0..A.len() {
                let k_const = Constant(F::from(k as u64));
                let eq_k = gate.is_equal(ctx, idx, k_const);
                A_idx = gate.select(ctx, A[k], A_idx, eq_k);
            }

            let eq = gate.is_equal(ctx, B[i][j], A_idx);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
