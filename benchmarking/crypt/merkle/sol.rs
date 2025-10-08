use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions};
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
    pub leaves: Vec<u64>,
    pub leaf_idx: usize,
    pub path: Vec<String>,
    pub bits: Vec<u64>,
}

// ---------------------------------------------------------
// Zinnia chips → Halo2 components
// ---------------------------------------------------------

fn mimc3<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    x: AssignedValue<F>,
    k: AssignedValue<F>,
) -> AssignedValue<F> {
    let mut t = gate.add(ctx, x, k);
    let consts = [1u64, 2, 3, 4, 5, 6, 7, 8];
    for c in consts {
        let c_val = Constant(F::from(c));
        let s = gate.add(ctx, t, c_val);
        let s2 = gate.mul(ctx, s, s);
        t = gate.mul(ctx, s2, s);
    }
    t
}

fn mimc_hash2<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    left: AssignedValue<F>,
    right: AssignedValue<F>,
) -> AssignedValue<F> {
    let zero = ctx.load_constant(F::ZERO);
    let sum = gate.add(ctx, left, right);
    mimc3(ctx, gate, sum, zero)
}

// ---------------------------------------------------------
// Merkle tree utilities
// ---------------------------------------------------------

fn merkle_root<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    leaves: &Vec<AssignedValue<F>>,
) -> AssignedValue<F> {
    // Level 0 → Level 1
    let mut L1: Vec<AssignedValue<F>> = Vec::new();
    for i in (0..8).step_by(2) {
        L1.push(mimc_hash2(ctx, gate, leaves[i], leaves[i + 1]));
    }

    // Level 1 → Level 2
    let mut L2: Vec<AssignedValue<F>> = Vec::new();
    for i in (0..4).step_by(2) {
        L2.push(mimc_hash2(ctx, gate, L1[i], L1[i + 1]));
    }

    // Level 2 → Root
    mimc_hash2(ctx, gate, L2[0], L2[1])
}

fn merkle_verify<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    leaf: AssignedValue<F>,
    path: &Vec<AssignedValue<F>>,
    bits: &Vec<AssignedValue<F>>,
    root: AssignedValue<F>,
) {
    let mut cur = leaf;
    for d in 0..3 {
        let is_left = gate.is_equal(ctx, bits[d], Constant(F::ZERO));
        let left = mimc_hash2(ctx, gate, cur, path[d]);
        let right = mimc_hash2(ctx, gate, path[d], cur);
        cur = gate.select(ctx, left, right, is_left);
    }
    let eq = gate.is_equal(ctx, cur, root);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

// ---------------------------------------------------------
// Main verification circuit
// ---------------------------------------------------------

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

    // Load leaves
    let leaves: Vec<AssignedValue<F>> = input
        .leaves
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // Load path elements
    let path: Vec<AssignedValue<F>> = input
        .path
        .iter()
        .map(|s| ctx.load_witness(F::from_str_vartime(s).unwrap()))
        .collect();

    // Load bits
    let bits: Vec<AssignedValue<F>> = input
        .bits
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();

    // Compute root from leaves
    let root = merkle_root(ctx, &gate, &leaves);

    // Select leaf[leaf_idx]
    let mut leaf_val = ctx.load_constant(F::from(0));
    for (i, leaf) in leaves.iter().enumerate() {
        let eq = gate.is_equal(ctx, Constant(F::from(i as u64)), Constant(F::from(input.leaf_idx as u64)));
        leaf_val = gate.select(ctx, *leaf, leaf_val, eq);
    }

    // Verify inclusion path
    merkle_verify(ctx, &gate, leaf_val, &path, &bits, root);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
