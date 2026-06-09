// DS-1000 case 339 — Halo2 baseline.
// Kernel: `np.vstack((np.diag(b), np.diag(np.fliplr(b))))` where
// `b = a[:8, :8]` (the leading 8×8 of a Float[8,10]).
// Mirrors Zinnia IR after DCE: 16× ReadFloatIR + 16× ExposePublicFIR.
// Only 16 of the 80 cells are consumed (diag + anti-diag of leading 8x8);
// Zinnia v7 DCE eliminates the dead ReadFloatIR ops, so HW skips them too.

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
pub struct CircuitInput { pub a: Vec<f64> }  // 80 cells (only 16 consumed)

fn diag_stack<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 80);
    // Only the 16 diag + anti-diag cells of b = a[:8,:8] are consumed; the
    // other 64 ReadFloatIR ops are eliminated by Zinnia v7 DCE, so HW skips
    // them too.
    // diag(b) for b = a[:8, :8]: b[i][i] == a[i * 10 + i] in the original [8,10] layout.
    for i in 0..8usize {
        let cell = ctx.load_witness(fixed_point_chip.quantization(input.a[i * 10 + i]));
        make_public.push(cell);
    }
    // diag(fliplr(b)): b[i][7 - i] == a[i * 10 + (7 - i)].
    for i in 0..8usize {
        let cell = ctx.load_witness(fixed_point_chip.quantization(input.a[i * 10 + (7 - i)]));
        make_public.push(cell);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(diag_stack, args); }
