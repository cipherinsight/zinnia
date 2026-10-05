// DS-1000 case 329 — Halo2 baseline (apples-to-apples with Zinnia).
//
// Kernel: result = a ** power, with `power: Integer` (runtime).
//
// Mirrors Zinnia's IR exactly (341 stmts):
//   1 ReadIntegerIR       (power)
//   1 FloatCastIR         (power int → fp)
//   1 ConstantFloatIR     (the constant 0.0 used in NotEqualFIR)
//  48 ReadFloatIR         (one per cell)
//  48 PowFIR              (qpow per cell, runtime exponent)
//  48 NotEqualFIR         (base != 0 — defensive)
//  48 GreaterThanOrEqualFIR (power >= 0)
//  48 LogicalOrIR         (a_ne_zero OR power_ge_zero)
//  48 AssertIR            (assert the OR — guards 0^negative)
//  48 ExposePublicFIR     (expose result per cell)
//
// No claim assertion — results are exposed as public outputs.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{
    AssignedValue,
    QuantumCell::Constant,
};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,      // 48 floats
    pub power: String,    // ReadIntegerIR: signed decimal integer
}

fn pow_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 48);

    // ReadIntegerIR + FloatCastIR for `power`.
    let power_abs = ctx.load_witness(F::from_u128(
        input.power.trim_start_matches('-')
            .parse::<u128>()
            .expect("power must be a decimal integer"),
    ));
    let power_int = if input.power.starts_with('-') {
        gate.neg(ctx, power_abs)
    } else {
        power_abs
    };
    let one_fp = ctx.load_witness(fixed_point_chip.quantization(1.0));
    let power_fp = fixed_point_chip.qmul(ctx, power_int, one_fp);

    // Shared constants for the safety check (base != 0 || power >= 0).
    let zero_fp = ctx.load_witness(fixed_point_chip.quantization(0.0));
    let eps_pos = Constant(fixed_point_chip.quantization(0.001));
    let eps_neg = Constant(fixed_point_chip.quantization(-0.001));

    for k in 0..48usize {
        // ReadFloatIR.
        let a_k = ctx.load_witness(fixed_point_chip.quantization(input.a[k]));

        // PowFIR.
        let result_k = fixed_point_chip.qpow(ctx, a_k, power_fp);

        // NotEqualFIR (a_k != 0): float-equality pattern, then negate.
        let diff = fixed_point_chip.qsub(ctx, a_k, zero_fp);
        let lt_pos = range_chip.is_less_than(ctx, diff, eps_pos, 128);
        let gt_neg = range_chip.is_less_than(ctx, eps_neg, diff, 128);
        let eq_zero = gate.and(ctx, lt_pos, gt_neg);
        let a_ne_zero = gate.not(ctx, eq_zero);

        // GreaterThanOrEqualFIR (power_int >= 0): !(power_int < 0).
        let zero_int = Constant(F::ZERO);
        let pow_lt_zero = range_chip.is_less_than(ctx, power_int, zero_int, 128);
        let pow_ge_zero = gate.not(ctx, pow_lt_zero);

        // LogicalOrIR.
        let safe = gate.or(ctx, a_ne_zero, pow_ge_zero);

        // AssertIR.
        gate.assert_is_const(ctx, &safe, &F::ONE);

        // ExposePublicFIR.
        make_public.push(result_k);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(pow_circuit, args);
}
