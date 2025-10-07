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
    pub A: Vec<Vec<u64>>,
    pub B: Vec<Vec<u64>>,
    pub output: Vec<Vec<u64>>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    // === initialize all chips before main context ===
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let _poseidon = PoseidonHasher::<F, T, RATE>::new(
        OptimizedPoseidonSpec::new::<R_F, R_P, 0>(),
    );

    // === now obtain main region ===
    let ctx = builder.main(0);

    // --- Load input matrices ---
    let mut A: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.A.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.A[i].len() {
            row.push(ctx.load_witness(F::from(input.A[i][j])));
        }
        A.push(row);
    }

    let mut B: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.B.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.B[i].len() {
            row.push(ctx.load_witness(F::from(input.B[i][j])));
        }
        B.push(row);
    }

    let mut output: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.output.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.output[i].len() {
            row.push(ctx.load_witness(F::from(input.output[i][j])));
        }
        output.push(row);
    }

    // --- Step 1: membership check ---
    let mut in_B: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..4 {
        let mut found = ctx.load_constant(F::ZERO);
        for j in 0..7 {
            let m0 = gate.is_equal(ctx, A[i][0], B[j][0]);
            let m1 = gate.is_equal(ctx, A[i][1], B[j][1]);
            let m2 = gate.is_equal(ctx, A[i][2], B[j][2]);
            let and01 = gate.and(ctx, m0, m1);
            let row_match = gate.and(ctx, and01, m2);
            found = gate.or(ctx, found, row_match);
        }
        in_B.push(found);
    }

    // --- Step 2: prefix counts & flags ---
    let mut pref = ctx.load_constant(F::ZERO);
    let mut pref_before: Vec<AssignedValue<F>> = Vec::new();
    let mut keep_flag: Vec<AssignedValue<F>> = Vec::new();

    for i in 0..4 {
        pref_before.push(pref);
        // not_in = 1 - in_B[i]
        let not_in = gate.not(ctx, in_B[i]);
        keep_flag.push(not_in);
        pref = gate.add(ctx, pref, not_in);
    }

    // --- Step 3: assert total kept == 2 ---
    let two = Constant(F::from(2));
    let eq_kept = gate.is_equal(ctx, pref, two);
    gate.assert_is_const(ctx, &eq_kept, &F::ONE);

    // --- Step 4: build expected kept rows ---
    let mut exp: Vec<Vec<AssignedValue<F>>> = vec![
        vec![ctx.load_constant(F::ZERO); 3],
        vec![ctx.load_constant(F::ZERO); 3],
    ];

    for i in 0..4 {
        let is_keep = keep_flag[i];
        let is_pos0 = gate.is_equal(ctx, pref_before[i], Constant(F::ZERO));
        let is_pos1 = gate.is_equal(ctx, pref_before[i], Constant(F::from(1u64)));

        let w0 = gate.and(ctx, is_keep, is_pos0);
        let w1 = gate.and(ctx, is_keep, is_pos1);

        for c in 0..3 {
            let a_ic = A[i][c];
            let add0 = gate.mul(ctx, a_ic, w0);
            let add1 = gate.mul(ctx, a_ic, w1);
            exp[0][c] = gate.add(ctx, exp[0][c], add0);
            exp[1][c] = gate.add(ctx, exp[1][c], add1);
        }
    }

    // --- Step 5: assert equality with output ---
    for r in 0..2 {
        for c in 0..3 {
            let eq = gate.is_equal(ctx, exp[r][c], output[r][c]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
