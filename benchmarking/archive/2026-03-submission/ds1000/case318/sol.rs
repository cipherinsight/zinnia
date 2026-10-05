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
    pub permutation: Vec<u64>,
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
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load a (2×5)
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    // Load permutation
    let permutation: Vec<AssignedValue<F>> = input
        .permutation
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // Load result (2×5)
    let mut result: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.result.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.result[i].len() {
            row.push(ctx.load_witness(F::from(input.result[i][j])));
        }
        result.push(row);
    }

    // === Main logic ===
    let n = 5;

    for j in 0..n {
        // --- Step 1: compute c[j] = Σ_i i * [permutation[i] == j]
        let mut cj = ctx.load_constant(F::from(0));
        let j_const = Constant(F::from(j as u64));

        for i in 0..n {
            let i_const = Constant(F::from(i as u64));
            let cond = gate.is_equal(ctx, permutation[i], j_const);
            let term = gate.mul(ctx, i_const, cond);
            cj = gate.add(ctx, cj, term);
        }

        // --- Step 2: select a[0, cj]
        let mut sel_val_r0 = ctx.load_constant(F::from(0));
        for t in 0..n {
            let t_const = Constant(F::from(t as u64));
            let cond = gate.is_equal(ctx, cj, t_const);
            let prod = gate.mul(ctx, a[0][t], cond);
            sel_val_r0 = gate.add(ctx, sel_val_r0, prod);
        }
        let eq0 = gate.is_equal(ctx, result[0][j], sel_val_r0);
        gate.assert_is_const(ctx, &eq0, &F::ONE);

        // --- Step 3: select a[1, cj]
        let mut sel_val_r1 = ctx.load_constant(F::from(0));
        for t in 0..n {
            let t_const = Constant(F::from(t as u64));
            let cond = gate.is_equal(ctx, cj, t_const);
            let prod = gate.mul(ctx, a[1][t], cond);
            sel_val_r1 = gate.add(ctx, sel_val_r1, prod);
        }
        let eq1 = gate.is_equal(ctx, result[1][j], sel_val_r1);
        gate.assert_is_const(ctx, &eq1, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
