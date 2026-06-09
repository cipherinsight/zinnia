// DS-1000 case 448 — Halo2 baseline.
// Kernel: `np.vstack(([x_dists.T], [y_dists.T])).T` on Float[3, 4] × Float[3, 4]
// → Float[3, 4, 2].
// Pure static gather: result[i][j][0] = x_dists[i][j]; result[i][j][1] = y_dists[i][j].
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x_dists: Vec<f64>, pub y_dists: Vec<f64> }

fn vstack_t<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.x_dists.len(), 12);
    assert_eq!(input.y_dists.len(), 12);
    let xs: Vec<AssignedValue<F>> = input.x_dists.iter().map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let ys: Vec<AssignedValue<F>> = input.y_dists.iter().map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    for i in 0..3 {
        for j in 0..4 {
            make_public.push(xs[i * 4 + j]);
            make_public.push(ys[i * 4 + j]);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(vstack_t, args); }
