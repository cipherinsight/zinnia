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
    pub result: Vec<u64>,
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

    // Load expected result
    let result: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    let n = 5;
    let last_col = n - 1;

    // flipped[i][j] = a[i][4 - j]
    let mut flipped: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..n {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..n {
            let j_const = Constant(F::from(j as u64));
            let rev_idx_const = Constant(F::from((last_col - j) as u64));

            // Select a[i][4 - j]
            let mut selected = ctx.load_constant(F::from(0));
            for k in 0..n {
                let k_const = Constant(F::from(k as u64));
                let eq = gate.is_equal(ctx, rev_idx_const, k_const);
                selected = gate.select(ctx, a[i][k], selected, eq);
            }
            row.push(selected);
        }
        flipped.push(row);
    }

    // diag_vals[k] = flipped[k][k]
    for k in 0..n {
        let mut selected = ctx.load_constant(F::from(0));
        for j in 0..n {
            let j_const = Constant(F::from(j as u64));
            let eq = gate.is_equal(ctx, Constant(F::from(k as u64)), j_const);
            selected = gate.select(ctx, flipped[k][j], selected, eq);
        }

        // Assert result[k] == flipped[k][k]
        let eq_diag = gate.is_equal(ctx, result[k], selected);
        gate.assert_is_const(ctx, &eq_diag, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
