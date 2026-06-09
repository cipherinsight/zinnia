// DS-1000 case 422 — Halo2 baseline.
// Kernel: np.correlate(a, np.hstack((b[1:], b)), 'valid') on Float[50]×Float[50].
// len(c) = 99, len(out) = 50. Formula: out[k] = sum_{n=0..49} a[n] * c[n + 49 - k].
// Each output is a length-50 inner product — use fpc.inner_product directly.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64>, pub b: Vec<f64> }

fn correlate_hstack<F: ScalarField>(
    bld: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(bld);
    let ctx = bld.main(0);
    assert_eq!(input.a.len(), 50);
    assert_eq!(input.b.len(), 50);

    let a_cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let b_cells: Vec<AssignedValue<F>> = input.b.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // c = hstack((b[1:], b)) → length 99: c[0..48] = b[1..49], c[49..98] = b[0..49].
    let c: Vec<AssignedValue<F>> = (0..99).map(|i| {
        if i < 49 { b_cells[i + 1] } else { b_cells[i - 49] }
    }).collect();

    // out[k] = sum_n a[n] * c[n + 49 - k] for k in 0..50.
    for k in 0..50usize {
        let shifted: Vec<AssignedValue<F>> = (0..50).map(|n| c[n + 49 - k]).collect();
        let dot = fpc.inner_product(ctx, a_cells.clone(), shifted);
        make_public.push(dot);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(correlate_hstack, args); }
