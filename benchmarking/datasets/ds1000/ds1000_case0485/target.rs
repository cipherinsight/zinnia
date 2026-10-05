// DS-1000 case 485 — Halo2 baseline.
// Kernel: `np.tensordot(A, B, axes=((2),(0)))` on Float[5,6,3] × Float[3,3].
// out[i,j,k] = sum_{l=0..3} A[i,j,l] * B[l,k]   — 90 outputs of length-3 dot.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,  // 90 floats (5×6×3), row-major
    pub b: Vec<f64>,  // 9 floats (3×3), row-major
}

fn tensordot<F: ScalarField>(
    bld: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(bld);
    let ctx = bld.main(0);

    assert_eq!(input.a.len(), 5 * 6 * 3);
    assert_eq!(input.b.len(), 3 * 3);

    let a: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let b: Vec<AssignedValue<F>> = input.b.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // For each (i, j, k) compute sum_l A[i,j,l] * B[l,k] via inner_product.
    for i in 0..5usize {
        for j in 0..6usize {
            let a_row: Vec<AssignedValue<F>> = (0..3).map(|l| a[i * 18 + j * 3 + l]).collect();
            for k in 0..3usize {
                let b_col: Vec<AssignedValue<F>> = (0..3).map(|l| b[l * 3 + k]).collect();
                let dot = fpc.inner_product(ctx, a_row.clone(), b_col);
                make_public.push(dot);
            }
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(tensordot, args); }
