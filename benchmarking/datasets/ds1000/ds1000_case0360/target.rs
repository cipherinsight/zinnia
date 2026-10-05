// DS-1000 case 360 — Halo2 baseline. `np.delete(a, 2, axis=0)` on Float[4, 4] → Float[3, 4].
// IR after DCE: 12× ReadF + 12× ExposeF.
// Only the 12 kept cells (rows 0, 1, 3) are consumed; Zinnia v7 DCE
// eliminates the dead ReadFloatIR ops for row 2, so HW skips them too.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 16 cells (only 12 consumed)

fn del_row2_4x4<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 16);
    // Only rows 0, 1, 3 are consumed; the row 2 ReadFloatIR ops are
    // eliminated by Zinnia v7 DCE, so HW skips them too.
    for i in [0usize, 1, 3] {
        for j in 0..4 {
            let cell = ctx.load_witness(fpc.quantization(input.a[i * 4 + j]));
            make_public.push(cell);
        }
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(del_row2_4x4, args); }
