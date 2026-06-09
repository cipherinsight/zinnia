// DS-1000 case 317 — Halo2 baseline.
// Kernel: `np.array(a)` on Float[10, 20, 5] — identity copy of 1000 cells.
// Mirrors Zinnia IR: 1000× ReadFloatIR + 1000× ExposePublicFIR.

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
pub struct CircuitInput {
    pub a: Vec<f64>,  // 1000 floats
}

fn identity_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 1000);
    for v in &input.a {
        let cell = ctx.load_witness(fixed_point_chip.quantization(*v));
        make_public.push(cell);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(identity_circuit, args);
}
