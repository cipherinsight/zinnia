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
    pub a: Vec<Vec<u64>>,
    pub result: Vec<Vec<Vec<u64>>>,
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

    // ---- Step 1: load matrix a ----
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    // ---- Step 2: trim to (4,4) ----
    // we drop last column (index 4)
    let mut trimmed: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..4 {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..4 {
            row.push(a[i][j]);
        }
        trimmed.push(row);
    }

    // ---- Step 3: emulate blockify + transpose + flatten ----
    // Equivalent to numpy logic:
    // x.reshape(2,2,2,2).transpose((0,2,1,3)).reshape(4,2,2)
    // Resulting index pattern in row-major flatten:
    // [
    //   [[1,5],[2,6]],
    //   [[9,13],[10,14]],
    //   [[3,7],[4,8]],
    //   [[11,15],[12,16]]
    // ]
    //
    // In flat indices of 4x4 matrix:
    // block0: [ (0,0),(0,1),(1,0),(1,1) ]
    // block1: [ (0,2),(0,3),(1,2),(1,3) ]
    // block2: [ (2,0),(2,1),(3,0),(3,1) ]
    // block3: [ (2,2),(2,3),(3,2),(3,3) ]

    let index_map = vec![
        vec![vec![(0, 0), (0, 1)], vec![(1, 0), (1, 1)]],
        vec![vec![(0, 2), (0, 3)], vec![(1, 2), (1, 3)]],
        vec![vec![(2, 0), (2, 1)], vec![(3, 0), (3, 1)]],
        vec![vec![(2, 2), (2, 3)], vec![(3, 2), (3, 3)]],
    ];

    let mut computed: Vec<Vec<Vec<AssignedValue<F>>>> =
        vec![vec![vec![ctx.load_constant(F::ZERO); 2]; 2]; 4];

    for b in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                let (r, c) = index_map[b][i][j];
                computed[b][i][j] = trimmed[r][c];
            }
        }
    }

    // ---- Step 4: verify equality with expected result ----
    for b in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                let expected = ctx.load_witness(F::from(input.result[b][i][j]));
                let eq = gate.is_equal(ctx, computed[b][i][j], expected);
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
