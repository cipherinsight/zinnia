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
    pub a: Vec<Vec<u64>>,
    pub result: Vec<Vec<u64>>,
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

    // Load input matrix a
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    // Load expected result (2 × dim)
    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.result.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.result[i].len() {
            row.push(ctx.load_witness(F::from(input.result[i][j])));
        }
        result.push(row);
    }

    let nrows = input.a.len();
    let ncols = input.a[0].len();
    let dim = if nrows < ncols { nrows } else { ncols };
    let last_col = dim - 1;

    // --- Step 1: Extract submatrix b = a[:dim, :dim]
    let mut b: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..dim {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..dim {
            row.push(a[i][j]);
        }
        b.push(row);
    }

    // --- Step 2: Compute main diagonal
    let mut main_diag: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..dim {
        let mut selected = ctx.load_constant(F::from(0));
        for j in 0..dim {
            let j_const = Constant(F::from(j as u64));
            let eq = gate.is_equal(ctx, Constant(F::from(i as u64)), j_const);
            selected = gate.select(ctx, b[i][j], selected, eq);
        }
        main_diag.push(selected);
    }

    // --- Step 3: Compute flipped (horizontal flip)
    let mut flipped: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..dim {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..dim {
            let rev_idx_const = Constant(F::from((last_col - j) as u64));
            let mut selected = ctx.load_constant(F::from(0));
            for k in 0..dim {
                let k_const = Constant(F::from(k as u64));
                let eq = gate.is_equal(ctx, rev_idx_const, k_const);
                selected = gate.select(ctx, b[i][k], selected, eq);
            }
            row.push(selected);
        }
        flipped.push(row);
    }

    // --- Step 4: Compute anti-diagonal
    let mut anti_diag: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..dim {
        let mut selected = ctx.load_constant(F::from(0));
        for j in 0..dim {
            let j_const = Constant(F::from(j as u64));
            let eq = gate.is_equal(ctx, Constant(F::from(i as u64)), j_const);
            selected = gate.select(ctx, flipped[i][j], selected, eq);
        }
        anti_diag.push(selected);
    }

    // --- Step 5: Verify stacked diagonals
    for j in 0..dim {
        let eq_main = gate.is_equal(ctx, result[0][j], main_diag[j]);
        gate.assert_is_const(ctx, &eq_main, &F::ONE);

        let eq_anti = gate.is_equal(ctx, result[1][j], anti_diag[j]);
        gate.assert_is_const(ctx, &eq_anti, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
