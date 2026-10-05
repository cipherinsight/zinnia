// DS-1000 case 385 — Halo2 baseline.
// Kernel: a.reshape(50, 2, 100, 2).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
// on Float[100, 200]. Output is (5000, 2, 2) = 20000 floats.
// Index mapping: for output[k][p][q] with k = c_block*50 + r_block,
//   source = a[r_block*2 + p][c_block*2 + q]
// The transpose swaps the block iteration: outer = c_block, inner = r_block.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,  // 100*200 = 20000 floats, row-major
}

fn patch_transpose<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const H: usize = 100;
    const W: usize = 200;
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), H * W);

    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // Block sweep with transposed order: c_block outer, r_block inner.
    for c_block in 0..(W / 2) {
        for r_block in 0..(H / 2) {
            for p in 0..2 {
                for q in 0..2 {
                    let row = r_block * 2 + p;
                    let col = c_block * 2 + q;
                    make_public.push(cells[row * W + col]);
                }
            }
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(patch_transpose, args);
}
