// DS-1000 case 439 — Halo2 baseline.
// Kernel: result = X.T[:, :, None] * X.T[:, None] on Float[10, 20] → Float[20, 10, 10].
// result[k][i][j] = X[i][k] * X[j][k]. Nested loops + qmul.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x: Vec<f64> }

fn outer_prod<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const ROWS: usize = 10;
    const COLS: usize = 20;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.x.len(), ROWS * COLS);

    let cells: Vec<AssignedValue<F>> = input.x.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let x = |i: usize, k: usize| cells[i * COLS + k];

    // result[k][i][j] = X[i][k] * X[j][k]; emit in k, i, j row-major.
    for k in 0..COLS {
        for i in 0..ROWS {
            for j in 0..ROWS {
                let p = fpc.qmul(ctx, x(i, k), x(j, k));
                make_public.push(p);
            }
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(outer_prod, args); }
