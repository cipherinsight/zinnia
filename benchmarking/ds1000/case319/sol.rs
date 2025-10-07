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
    pub a: Vec<Vec<Vec<u64>>>,     // shape (3,2,2)
    pub permutation: Vec<u64>,     // shape (3,)
    pub result: Vec<Vec<Vec<u64>>> // shape (3,2,2)
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

    // --- Load 3D array a ---
    let mut a: Vec<Vec<Vec<AssignedValue<F>>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut mat: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for r in 0..input.a[i].len() {
            let mut row: Vec<AssignedValue<F>> = Vec::new();
            for s in 0..input.a[i][r].len() {
                row.push(ctx.load_witness(F::from(input.a[i][r][s])));
            }
            mat.push(row);
        }
        a.push(mat);
    }

    // --- Load permutation ---
    let permutation: Vec<AssignedValue<F>> = input
        .permutation
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // --- Load result (3×2×2) ---
    let mut result: Vec<Vec<Vec<AssignedValue<F>>>> = Vec::new();
    for i in 0..input.result.len() {
        let mut mat: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for r in 0..input.result[i].len() {
            let mut row: Vec<AssignedValue<F>> = Vec::new();
            for s in 0..input.result[i][r].len() {
                row.push(ctx.load_witness(F::from(input.result[i][r][s])));
            }
            mat.push(row);
        }
        result.push(mat);
    }

    // === Core logic ===
    let n = 3;

    for k in 0..n {
        // --- Step 1: build inverse index c[k] = Σ_i i * [permutation[i] == k]
        let mut ck = ctx.load_constant(F::from(0));
        let k_const = Constant(F::from(k as u64));

        for i in 0..n {
            let i_const = Constant(F::from(i as u64));
            let cond = gate.is_equal(ctx, permutation[i], k_const);
            let term = gate.mul(ctx, i_const, cond);
            ck = gate.add(ctx, ck, term);
        }

        // --- Step 2: for each (r,s), select a[ck, r, s] ---
        for r in 0..2 {
            for s in 0..2 {
                let mut selected = ctx.load_constant(F::from(0));
                for t in 0..n {
                    let t_const = Constant(F::from(t as u64));
                    let cond = gate.is_equal(ctx, ck, t_const);
                    let prod = gate.mul(ctx, a[t][r][s], cond);
                    selected = gate.add(ctx, selected, prod);
                }

                // --- Step 3: enforce equality ---
                let eq = gate.is_equal(ctx, result[k][r][s], selected);
                gate.assert_is_const(ctx, &eq, &F::ONE);
            }
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
