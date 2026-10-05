// DS-1000 case 413 — Halo2 baseline. `x[x.imag != 0]` on Float[10].
// For real inputs x.imag is always 0, so the mask is empty and the result
// is empty. Zinnia compiles to 10× ReadFloat + 0× Expose (no output).
// We mirror that: witness the 10 inputs but expose nothing.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x: Vec<f64> }

fn imag_filter<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, _make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.x.len(), 10);
    // `x` is unused by the kernel; Zinnia v7 DCE-eliminates the dead
    // ReadFloatIR ops, so HW skips the loads too.
    let _ = (ctx, fpc);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(imag_filter, args); }
