// DS-1000 case 337 — Halo2 baseline.
// Kernel: `np.diag(np.fliplr(a))` on Float[8, 10] — 8 anti-diagonal cells.
// Mirrors Zinnia IR after DCE: 8× ReadFloatIR + 8× ExposePublicFIR.
// Pure static-index gather: a[i][9 - i] for i in 0..8.
// Only 8 of the 80 cells are consumed; Zinnia v7 DCE eliminates the dead
// ReadFloatIR ops, so HW skips the loads too.

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
pub struct CircuitInput { pub a: Vec<f64> }  // 80 cells, row-major (only 8 consumed)

fn antidiag_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 80);
    // Only the 8 anti-diagonal cells are consumed; the other 72 ReadFloatIR
    // ops are eliminated by Zinnia v7 DCE, so HW skips them too.
    for i in 0..8usize {
        let idx = i * 10 + (9 - i);
        let cell = ctx.load_witness(fixed_point_chip.quantization(input.a[idx]));
        make_public.push(cell);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(antidiag_circuit, args); }
