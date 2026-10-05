// DS-1000 case 472 — Halo2 baseline.
// Kernel: `a[-1:, ...]` on Float[6, 2, 9, 3, 7, 4] → Float[1, 2, 9, 3, 7, 4].
// Pure gather of the last slab (1512 floats) along axis 0 — layout-only.
// Only the last 1512 of 9072 cells are consumed; Zinnia v7 DCE eliminates
// the dead ReadFloatIR ops for the leading 5 slabs, so HW skips them too.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }  // 9072 floats, row-major (only 1512 consumed)

fn slab_last<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);

    const N: usize = 6 * 2 * 9 * 3 * 7 * 4; // 9072
    const SLAB: usize = 2 * 9 * 3 * 7 * 4;  // 1512
    assert_eq!(input.a.len(), N);

    // Last slab along axis 0: input.a[5*SLAB ..]. Only those 1512 cells are
    // consumed; the leading 7560 ReadFloatIR ops are eliminated by Zinnia v7
    // DCE, so HW skips them too.
    let start = 5 * SLAB;
    for k in 0..SLAB {
        let cell = ctx.load_witness(fpc.quantization(input.a[start + k]));
        make_public.push(cell);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(slab_last, args); }
