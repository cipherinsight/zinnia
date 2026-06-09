// DS-1000 case 334 — Halo2 baseline (apples-to-apples with Zinnia).
//
// Kernel: result = np.mean([a, b, c], axis=0)
//
// Mirrors Zinnia's IR exactly (355 stmts):
//  150 ReadFloatIR        (50 each for a, b, c)
//  100 AddFIR             (2 adds per cell: (a+b)+c)
//   50 DivFIR             (divide sum by 3 to get mean)
//    1 ConstantFloatIR    (the constant 3.0)
//    ... (plus 3 AllocateMemoryIR / 1 ConstantBoolIR for input metadata)
//   50 ExposePublicFIR    (expose 50 means)
//
// No claim assertion — Zinnia just exposes 50 floats.

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
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
}

fn mean_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 50);
    assert_eq!(input.b.len(), 50);
    assert_eq!(input.c.len(), 50);

    // ConstantFloatIR for 3.0 (shared across all cells).
    let three = ctx.load_witness(fixed_point_chip.quantization(3.0));

    for k in 0..50usize {
        let a_k = ctx.load_witness(fixed_point_chip.quantization(input.a[k]));
        let b_k = ctx.load_witness(fixed_point_chip.quantization(input.b[k]));
        let c_k = ctx.load_witness(fixed_point_chip.quantization(input.c[k]));

        // (a + b) + c, then / 3.
        let ab = fixed_point_chip.qadd(ctx, a_k, b_k);
        let sum = fixed_point_chip.qadd(ctx, ab, c_k);
        let mean = fixed_point_chip.qdiv(ctx, sum, three);

        // ExposePublicFIR.
        make_public.push(mean);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(mean_circuit, args);
}
