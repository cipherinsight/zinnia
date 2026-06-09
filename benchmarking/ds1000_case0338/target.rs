// DS-1000 case 338 — Halo2 baseline.
// Kernel: `np.vstack((np.diag(a), np.diag(np.fliplr(a))))` on Float[8, 8].
// Output Float[2, 8]: row 0 = diag, row 1 = anti-diag.
// IR after DCE: 16× ReadF + 16× ExposeF.
// Only 16 of the 64 cells are consumed (diag + anti-diag); Zinnia v7 DCE
// eliminates the dead ReadFloatIR ops, so HW skips the loads too.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 64 cells (only 16 consumed)

fn diag_vstack_8<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 64);
    // Only the 16 diag + anti-diag cells are consumed; the other 48 ReadFloatIR
    // ops are eliminated by Zinnia v7 DCE, so HW skips them too.
    // diag(a): a[i][i] for i in 0..8.
    for i in 0..8usize {
        let cell = ctx.load_witness(fpc.quantization(input.a[i * 8 + i]));
        make_public.push(cell);
    }
    // diag(fliplr(a)): a[i][7-i] for i in 0..8.
    for i in 0..8usize {
        let cell = ctx.load_witness(fpc.quantization(input.a[i * 8 + (7 - i)]));
        make_public.push(cell);
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(diag_vstack_8, args); }
