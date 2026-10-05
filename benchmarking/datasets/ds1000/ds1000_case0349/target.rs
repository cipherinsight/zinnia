// DS-1000 case 349 — Halo2 baseline.
// Kernel: `a.shape[1]` on Float[8, 6] → scalar integer 6.
// `a` is never read by the kernel; Zinnia v7 DCE eliminates the dead
// witness loads, so HW shouldn't carry them either (otherwise the
// comparison is artificially biased against HW).

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
pub struct CircuitInput { pub a: Vec<f64> }  // 48 cells (unused)

fn shape1_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let _fpc = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 48);
    // `a` is unused by the kernel; no witnesses loaded (matches Zinnia v7
    // DCE-eliminated emission).
    let six = ctx.load_witness(F::from(6u64));
    make_public.push(six);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(shape1_circuit, args); }
