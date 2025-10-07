use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::Constant,
};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use serde::{Serialize, Deserialize};
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;
use halo2_base::poseidon::hasher::PoseidonHasher;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<Vec<u64>>,               // 4 x 4
    pub result: Vec<Vec<Vec<u64>>>,     // 4 x 2 x 2
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
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

    // Load a[4][4]
    let mut a: Vec<Vec<AssignedValue<F>>> = Vec::new();
    for i in 0..input.a.len() {
        let mut row: Vec<AssignedValue<F>> = Vec::new();
        for j in 0..input.a[i].len() {
            row.push(ctx.load_witness(F::from(input.a[i][j])));
        }
        a.push(row);
    }

    // Load result[4][2][2]
    let mut result: Vec<Vec<Vec<AssignedValue<F>>>> = Vec::new();
    for o in 0..input.result.len() {
        let mut plane: Vec<Vec<AssignedValue<F>>> = Vec::new();
        for r in 0..input.result[o].len() {
            let mut row: Vec<AssignedValue<F>> = Vec::new();
            for c in 0..input.result[o][r].len() {
                row.push(ctx.load_witness(F::from(input.result[o][r][c])));
            }
            plane.push(row);
        }
        result.push(plane);
    }

    // Assert: result[o][r][c] == a[i][j], with
    // o = floor(j/2)*2 + floor(i/2), r = i%2, c = j%2
    for i in 0..4 {
        for j in 0..4 {
            let o = (j / 2) * 2 + (i / 2);
            let r = i % 2;
            let c = j % 2;

            let eq = gate.is_equal(ctx, result[o][r][c], a[i][j]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
