// DS-1000 case 323 — Halo2 baseline (apples-to-apples with Zinnia).
//
// Kernel: result = np.sin(np.deg2rad(degree))
//
// Mirrors Zinnia's IR exactly (6 stmts):
//   1 ReadIntegerIR       → load_witness(F::from_u128(...)) + optional neg
//   1 FloatCastIR         → qmul(int_val, quantized(1.0))
//   1 ConstantFloatIR     → quantized(π/180)
//   1 MulFIR              → qmul(rad_factor, deg_fp)
//   1 SinFIR              → fixed_point_chip.qsin (single primitive)
//   1 ExposePublicFIR     → make_public.push(sin_val)
//
// No claim assertion — Zinnia just exposes the result.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub degree: String,  // ReadIntegerIR: signed decimal integer
}

fn sin_deg2rad<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    // ReadIntegerIR: u128 parse + optional neg.
    let degree_abs = ctx.load_witness(F::from_u128(
        input.degree.trim_start_matches('-')
            .parse::<u128>()
            .expect("degree must be a decimal integer"),
    ));
    let degree_signed = if input.degree.starts_with('-') {
        gate.neg(ctx, degree_abs)
    } else {
        degree_abs
    };

    // FloatCastIR: promote integer to fp via qmul with quantised 1.0.
    let one_fp = ctx.load_witness(fixed_point_chip.quantization(1.0));
    let degree_fp = fixed_point_chip.qmul(ctx, degree_signed, one_fp);

    // ConstantFloatIR + MulFIR: rad = degree * π/180.
    let pi_over_180 = ctx.load_witness(fixed_point_chip.quantization(std::f64::consts::PI / 180.0));
    let rad = fixed_point_chip.qmul(ctx, degree_fp, pi_over_180);

    // SinFIR.
    let sin_val = fixed_point_chip.qsin(ctx, rad);

    // ExposePublicFIR.
    make_public.push(sin_val);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(sin_deg2rad, args);
}
