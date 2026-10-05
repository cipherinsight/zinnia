// DS-1000 case 361 — Halo2 baseline.
// Kernel: `np.delete(a, [0, 2], axis=1)` on Float[6, 6] → Float[6, 4].
// Mirrors Zinnia IR after DCE: 24× ReadFloatIR + 24× ExposePublicFIR.
// Static delete: cols 0 and 2 are dropped; remaining cols 1, 3, 4, 5.
// Only 24 of the 36 cells are consumed; Zinnia v7 DCE eliminates the dead
// ReadFloatIR ops for cols 0 and 2, so HW skips those loads too.

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
pub struct CircuitInput { pub a: Vec<f64> }  // 36 cells (only 24 consumed)

fn delete_static<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 36);
    // Output: a[i][1], a[i][3], a[i][4], a[i][5] for i in 0..6.
    // Only the kept cols are consumed; the col 0 and col 2 ReadFloatIR ops
    // are eliminated by Zinnia v7 DCE, so HW skips them too.
    let kept = [1usize, 3, 4, 5];
    for i in 0..6 {
        for &j in &kept {
            let cell = ctx.load_witness(fixed_point_chip.quantization(input.a[i * 6 + j]));
            make_public.push(cell);
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(delete_static, args); }
