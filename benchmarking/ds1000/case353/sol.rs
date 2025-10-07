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

    // ---- Load A, B, output ----
    let mut A: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.A.len() {
        A.push(
            input.A[i]
                .iter()
                .map(|x| ctx.load_witness(F::from(*x)))
                .collect(),
        );
    }

    let mut B: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.B.len() {
        B.push(
            input.B[i]
                .iter()
                .map(|x| ctx.load_witness(F::from(*x)))
                .collect(),
        );
    }

    let mut output: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.output.len() {
        output.push(
            input.output[i]
                .iter()
                .map(|x| ctx.load_witness(F::from(*x)))
                .collect(),
        );
    }

    // ---- Step 1: Membership check (in_B[i]) ----
    let n_a = A.len();
    let n_b = B.len();
    let mut in_B: Vec<AssignedValue<F>> = Vec::new();

    for i in 0..n_a {
        let mut found = ctx.load_constant(F::ZERO);
        for j in 0..n_b {
            let m0 = gate.is_equal(ctx, A[i][0], B[j][0]);
            let m1 = gate.is_equal(ctx, A[i][1], B[j][1]);
            let m2 = gate.is_equal(ctx, A[i][2], B[j][2]);
            let row_match = gate.and(ctx, gate.and(ctx, m0, m1), m2);
            found = gate.or(ctx, found, row_match);
        }
        in_B.push(found);
    }

    // ---- Step 2: prefix count for rows NOT in B ----
    let mut pref = ctx.load_constant(F::ZERO);
    let mut pref_before: Vec<AssignedValue<F>> = Vec::new();
    let mut keep_flag: Vec<AssignedValue<F>> = Vec::new();

    for i in 0..n_a {
        pref_before.push(pref);
        // not_in = 1 - in_B[i]
        let not_in = gate.not(ctx, in_B[i]);
        keep_flag.push(not_in);
        pref = gate.add(ctx, pref, not_in);
    }

    // assert pref == 2
    let eq_pref = gate.is_equal(ctx, pref, Constant(F::from(2)));
    gate.assert_is_const(ctx, &eq_pref, &F::ONE);

    // ---- Step 3: build expected kept rows ----
    let mut exp: Vec<Vec<AssignedValue<F>>> =
        vec![vec![ctx.load_constant(F::ZERO); 3], vec![ctx.load_constant(F::ZERO); 3]];

    for i in 0..n_a {
        let is_keep = keep_flag[i];

        // is_pos0 = (pref_before[i] == 0)
        let is_pos0 = gate.is_equal(ctx, pref_before[i], Constant(F::ZERO));
        // is_pos1 = (pref_before[i] == 1)
        let is_pos1 = gate.is_equal(ctx, pref_before[i], Constant(F::from(1)));

        // w0 = is_keep * is_pos0
        let w0 = gate.mul(ctx, is_keep, is_pos0);
        // w1 = is_keep * is_pos1
        let w1 = gate.mul(ctx, is_keep, is_pos1);

        for c in 0..3 {
            let add0 = gate.mul(ctx, A[i][c], w0);
            let add1 = gate.mul(ctx, A[i][c], w1);
            exp[0][c] = gate.add(ctx, exp[0][c], add0);
            exp[1][c] = gate.add(ctx, exp[1][c], add1);
        }
    }

    // ---- Step 4: Compare with output ----
    for r in 0..2 {
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
