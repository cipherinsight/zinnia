// DS-1000 case 409 — Halo2 baseline.
// Kernel: `z = x + y` for Float[8, 6] each.
// Mirrors Zinnia IR: 96× ReadFloatIR + 48× AddFIR + 48× ExposePublicFIR.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

fn add_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.x.len(), 48);
    assert_eq!(input.y.len(), 48);
    for k in 0..48 {
        let xk = ctx.load_witness(fixed_point_chip.quantization(input.x[k]));
        let yk = ctx.load_witness(fixed_point_chip.quantization(input.y[k]));
        let zk = fixed_point_chip.qadd(ctx, xk, yk);
        make_public.push(zk);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(add_circuit, args);
}
