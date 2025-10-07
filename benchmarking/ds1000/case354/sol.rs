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
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load all matrices
    let mut A: Vec<Vec<AssignedValue<F>>> = input
        .A
        .iter()
        .map(|row| row.iter().map(|x| ctx.load_witness(F::from(*x))).collect())
        .collect();

    let mut B: Vec<Vec<AssignedValue<F>>> = input
        .B
        .iter()
        .map(|row| row.iter().map(|x| ctx.load_witness(F::from(*x))).collect())
        .collect();

    let mut output: Vec<Vec<AssignedValue<F>>> = input
        .output
        .iter()
        .map(|row| row.iter().map(|x| ctx.load_witness(F::from(*x))).collect())
        .collect();

    let n_a = A.len();
    let n_b = B.len();

    // ---- Step 1: membership flags ----
    let mut inB: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..n_a {
        let mut found = ctx.load_constant(F::ZERO);
        for j in 0..n_b {
            let m0 = gate.is_equal(ctx, A[i][0], B[j][0]);
            let m1 = gate.is_equal(ctx, A[i][1], B[j][1]);
            let m2 = gate.is_equal(ctx, A[i][2], B[j][2]);
            let row_match = gate.and(ctx, gate.and(ctx, m0, m1), m2);
            found = gate.or(ctx, found, row_match);
        }
        inB.push(found);
    }

    let mut inA: Vec<AssignedValue<F>> = Vec::new();
    for j in 0..n_b {
        let mut found = ctx.load_constant(F::ZERO);
        for i in 0..n_a {
            let m0 = gate.is_equal(ctx, B[j][0], A[i][0]);
            let m1 = gate.is_equal(ctx, B[j][1], A[i][1]);
            let m2 = gate.is_equal(ctx, B[j][2], A[i][2]);
            let row_match = gate.and(ctx, gate.and(ctx, m0, m1), m2);
            found = gate.or(ctx, found, row_match);
        }
        inA.push(found);
    }

    // ---- Step 2: prefix counts for A-only and B-only ----
    // A-side
    let mut prefA = ctx.load_constant(F::ZERO);
    let mut prefA_before: Vec<AssignedValue<F>> = Vec::new();
    let mut keepA: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..n_a {
        prefA_before.push(prefA);
        let not_inB = gate.not(ctx, inB[i]);
        keepA.push(not_inB);
        prefA = gate.add(ctx, prefA, not_inB);
    }
    let eq_prefA = gate.is_equal(ctx, prefA, Constant(F::from(2)));
    gate.assert_is_const(ctx, &eq_prefA, &F::ONE);

    // B-side
    let mut prefB = ctx.load_constant(F::ZERO);
    let mut prefB_before: Vec<AssignedValue<F>> = Vec::new();
    let mut keepB: Vec<AssignedValue<F>> = Vec::new();
    for j in 0..n_b {
        prefB_before.push(prefB);
        let not_inA = gate.not(ctx, inA[j]);
        keepB.push(not_inA);
        prefB = gate.add(ctx, prefB, not_inA);
    }
    let eq_prefB = gate.is_equal(ctx, prefB, Constant(F::from(5)));
    gate.assert_is_const(ctx, &eq_prefB, &F::ONE);

    // ---- Step 3: construct expected symmetric difference ----
    let mut exp: Vec<Vec<AssignedValue<F>>> =
        vec![vec![ctx.load_constant(F::ZERO); 3]; 7];

    // A-only rows
    for i in 0..n_a {
        let is_keep = keepA[i];
        let is_pos0 = gate.is_equal(ctx, prefA_before[i], Constant(F::ZERO));
        let is_pos1 = gate.is_equal(ctx, prefA_before[i], Constant(F::from(1)));
        let w0 = gate.mul(ctx, is_keep, is_pos0);
        let w1 = gate.mul(ctx, is_keep, is_pos1);

        for c in 0..3 {
            let add0 = gate.mul(ctx, A[i][c], w0);
            let add1 = gate.mul(ctx, A[i][c], w1);
            exp[0][c] = gate.add(ctx, exp[0][c], add0);
            exp[1][c] = gate.add(ctx, exp[1][c], add1);
        }
    }

    // B-only rows
    for j in 0..n_b {
        let is_keep = keepB[j];
        for r in 0..5 {
            let at_r = gate.is_equal(ctx, prefB_before[j], Constant(F::from(r as u64)));
            let w = gate.mul(ctx, is_keep, at_r);
            for c in 0..3 {
                let add = gate.mul(ctx, B[j][c], w);
                exp[2 + r][c] = gate.add(ctx, exp[2 + r][c], add);
            }
        }
    }

    // ---- Step 4: compare with output ----
    for r in 0..7 {
        for c in 0..3 {
            let eq = gate.is_equal(ctx, output[r][c], exp[r][c]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
