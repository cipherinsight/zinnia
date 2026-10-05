// DS-1000 case 359 — Halo2 baseline. `np.delete(a, 2, axis=1)` on Float[3, 3] → Float[3, 2].
// Static delete: keep cols 0 and 1. IR after DCE: 6× ReadF + 6× ExposeF.
// Only the 6 kept cells are consumed; Zinnia v7 DCE eliminates the dead
// ReadFloatIR ops for col 2, so HW skips those loads too.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 9 cells (only 6 consumed)

fn del_col2_3x3<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 9);
    // Only cols 0 and 1 of each row are consumed; the col 2 ReadFloatIR ops
    // are eliminated by Zinnia v7 DCE, so HW skips them too.
    for i in 0..3 {
        for j in [0usize, 1] {
            let cell = ctx.load_witness(fpc.quantization(input.a[i * 3 + j]));
            make_public.push(cell);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(del_col2_3x3, args); }
