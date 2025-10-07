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
    pub x: Vec<Vec<u64>>,
    pub y: Vec<Vec<u64>>,
    pub z: Vec<Vec<u64>>,
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
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let nrows = input.x.len();
    let ncols = input.x[0].len();

    // --- Load inputs ---
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut z_vals = Vec::new();

    for i in 0..nrows {
        let mut x_row = Vec::new();
        let mut y_row = Vec::new();
        let mut z_row = Vec::new();
        for j in 0..ncols {
            x_row.push(ctx.load_witness(F::from(input.x[i][j])));
            y_row.push(ctx.load_witness(F::from(input.y[i][j])));
            z_row.push(ctx.load_witness(F::from(input.z[i][j])));
        }
        x_vals.push(x_row);
        y_vals.push(y_row);
        z_vals.push(z_row);
    }

    // --- Enforce z = x + y elementwise ---
    for i in 0..nrows {
        for j in 0..ncols {
            let sum = gate.add(ctx, x_vals[i][j], y_vals[i][j]);
            let eq = gate.is_equal(ctx, sum, z_vals[i][j]);
            gate.assert_is_const(ctx, &eq, &F::ONE);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
