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
    pub a: Vec<Vec<u64>>,       // 2×2 matrix
    pub result: Vec<Vec<u64>>,  // 2×2 matrix
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
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // --- Load input matrix a ---
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    // --- Load expected result ---
    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.result.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.result[i].len() {
            row.push(ctx.load_witness(F::from(input.result[i][j])));
        }
        result.push(row);
    }

    // Step 1: Compute min_val = a.min()
    let mut flat: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            flat.push(a[i][j]);
        }
    }

    let mut min_val = flat[0];
    for i in 1..flat.len() {
        let cur = flat[i];
        let less = range_chip.is_less_than(ctx, cur, min_val, 128);
        min_val = gate.select(ctx, cur, min_val, less);
    }

    // Step 2: Build expected = indices where a[i][j] == min_val
    // expected is a 2×2 zero matrix, and we fill rows where condition holds.
    let mut expected: Vec<Vec<AssignedValue<F>>> = vec![
        vec![ctx.load_constant(F::from(0)), ctx.load_constant(F::from(0))],
        vec![ctx.load_constant(F::from(0)), ctx.load_constant(F::from(0))]
    ];

    let mut idx = ctx.load_constant(F::from(0));
    for i in 0..2 {
        for j in 0..2 {
            let cond = gate.is_equal(ctx, a[i][j], min_val);

            // if cond: expected[idx, 0] = i; expected[idx, 1] = j
            for r in 0..2 {
                let idx_const = Constant(F::from(r as u64));
                let match_row = gate.is_equal(ctx, idx, idx_const);
                let active = gate.and(ctx, match_row, cond);

                let val_i = Constant(F::from(i as u64));
                let val_j = Constant(F::from(j as u64));

                let old0 = expected[r][0];
                let old1 = expected[r][1];
                expected[r][0] = gate.select(ctx, val_i, old0, active);
                expected[r][1] = gate.select(ctx, val_j, old1, active);
            }

            // idx += cond
            idx = gate.add(ctx, idx, cond);
        }
    }

    // Step 3: Verify equality: result == expected
    for i in 0..2 {
        for j in 0..2 {
            let eq = gate.is_equal(ctx, result[i][j], expected[i][j]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
