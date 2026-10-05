// DS-1000 case 500 — Halo2 baseline.
// Kernel: `a.reshape(-1, 3)` on Float[54] → Float[18, 3].
// Mirrors Zinnia IR: 54× ReadFloatIR + 54× ExposePublicFIR.
// Reshape is layout-only — same cells, exposed in row-major order.

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
    pub a: Vec<f64>,
}

fn reshape_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 54);
    for v in &input.a {
        let cell = ctx.load_witness(fixed_point_chip.quantization(*v));
        make_public.push(cell);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(reshape_circuit, args);
}
