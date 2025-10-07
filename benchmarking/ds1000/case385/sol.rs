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
    let _range_chip = builder.range_chip();
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

    // ---- Step 2: perform reshape(2,2,2,2) + transpose(0,2,1,3) + transpose(1,0,2,3) ----
    // We emulate this with explicit index arithmetic, equivalent to the Python ops.

    // Flatten a[4][4] into a vector of length 16 (row-major)
    let mut flat: Vec<AssignedValue<F>> = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            flat.push(a[i][j]);
        }
    }

    // Compute indices of final reshaped/transposed (4,2,2)
    // The mapping is determined by the NumPy transformation sequence.
    // Manual derivation yields the following block extraction pattern:
    // Block0: [[1,5],[2,6]]
    // Block1: [[3,7],[4,8]]
    // Block2: [[9,13],[10,14]]
    // Block3: [[11,15],[12,16]]
    // (indices in flat array correspond accordingly)
    let mut computed: Vec<Vec<Vec<AssignedValue<F>>>> = vec![vec![vec![ctx.load_constant(F::ZERO); 2]; 2]; 4];
    let index_map = vec![
        vec![vec![0, 1], vec![4, 5]],    // block0
        vec![vec![2, 3], vec![6, 7]],    // block1
        vec![vec![8, 9], vec![12, 13]],  // block2
        vec![vec![10, 11], vec![14, 15]] // block3
    ];

    for b in 0..4 {
        for i in 0..2 {
            for j in 0..2 {
                let idx = index_map[b][i][j];
                computed[b][i][j] = flat[idx];
            }
        }
    }

    // ---- Step 3: verify equality with expected result ----
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
