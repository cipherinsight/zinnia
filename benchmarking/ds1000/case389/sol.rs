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
    pub a: Vec<Vec<Vec<u64>>>,
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
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // ---- Load input tensor a (4×2×3) ----
    let mut a: Vec<Vec<Vec<AssignedValue<F>>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut block: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for r in 0..input.a[i].len() {
            let mut row: Vec<AssignedValue<F>> = Vec::new();
            for c in 0..input.a[i][r].len() {
                row.push(ctx.load_witness(F::from(input.a[i][r][c])));
            }
            block.push(row);
        }
        a.push(block);
    }

    // ---- Compute equivalent of reshape((2,2,2,3)) → transpose((0,2,1,3)) → reshape((4,6)) ----
    //
    // The Zinnia code produces a tiling:
    // result =
    // [[ 0,  1,  2,  3,  4,  5],
    //  [ 6,  7,  8,  9, 10, 11],
    //  [12, 13, 14, 15, 16, 17],
    //  [18, 19, 20, 21, 22, 23]]
    //
    // Mapping derived from block layout (row-major block tiling)
    // Each a[i] (2×3) fills:
    // block0 → top-left,
    // block1 → top-right,
    // block2 → bottom-left,
    // block3 → bottom-right.

    let index_map = vec![
        // block 0
        vec![
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
        ],
        // block 1 handled via continuity in pattern (flatten of 0,1,2,...)
    ];

    // Manually reproduce the expected flatten logic:
    // top half rows 0–1 from a[0], a[1]; bottom half rows 2–3 from a[2], a[3].
    let mut computed: Vec<Vec<AssignedValue<F>>> =
        vec![vec![ctx.load_constant(F::ZERO); 6]; 4];

    // top-left block (a[0]) → rows (0,1), cols (0..3)
    for r in 0..2 {
        for c in 0..3 {
            computed[r][c] = a[0][r][c];
        }
    }

    // top-right block (a[1]) → rows (0,1), cols (3..6)
    for r in 0..2 {
        for c in 0..3 {
            computed[r][c + 3] = a[1][r][c];
        }
    }

    // bottom-left block (a[2]) → rows (2,3), cols (0..3)
    for r in 0..2 {
        for c in 0..3 {
            computed[r + 2][c] = a[2][r][c];
        }
    }

    // bottom-right block (a[3]) → rows (2,3), cols (3..6)
    for r in 0..2 {
        for c in 0..3 {
            computed[r + 2][c + 3] = a[3][r][c];
        }
    }

    // ---- Compare computed with expected result ----
    for i in 0..4 {
        for j in 0..6 {
            let expected = ctx.load_witness(F::from(input.result[i][j]));
            let eq = gate.is_equal(ctx, computed[i][j], expected);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
