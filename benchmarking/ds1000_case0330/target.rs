// DS-1000 case 330 — Halo2 baseline.
// Kernel: result = a ** power, with a: Float[8, 6], power: Integer.
//
// High-level reimplementation: use fpc.qpow for elementwise power, with
// the standard 0^negative safety guard (a != 0 OR power >= 0).

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,    // 48 floats (8 x 6, row-major)
    pub power: String,  // signed integer
}

fn pow_elementwise<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 48);

    // Load power (signed integer → float for qpow).
    let power_abs = ctx.load_witness(F::from_u128(
        input.power.trim_start_matches('-').parse::<u128>().expect("decimal"),
    ));
    let power_int = if input.power.starts_with('-') { gate.neg(ctx, power_abs) } else { power_abs };
    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let power_fp = fpc.qmul(ctx, power_int, one_fp);

    // Shared constants for the 0^neg safety check.
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    let eps_pos = Constant(fpc.quantization(0.001));
    let eps_neg = Constant(fpc.quantization(-0.001));
    let zero_int = Constant(F::ZERO);

    for k in 0..48usize {
        let a_k = ctx.load_witness(fpc.quantization(input.a[k]));

        // result_k = a_k ** power_fp.
        let result_k = fpc.qpow(ctx, a_k, power_fp);

        // Safety: a_k != 0 OR power_int >= 0.
        let diff = fpc.qsub(ctx, a_k, zero_fp);
        let lt_p = range_chip.is_less_than(ctx, diff, eps_pos, 128);
        let gt_n = range_chip.is_less_than(ctx, eps_neg, diff, 128);
        let eq_zero = gate.and(ctx, lt_p, gt_n);
        let a_ne_zero = gate.not(ctx, eq_zero);

        let pow_lt_zero = range_chip.is_less_than(ctx, power_int, zero_int, 128);
        let pow_ge_zero = gate.not(ctx, pow_lt_zero);

        let safe = gate.or(ctx, a_ne_zero, pow_ge_zero);
        gate.assert_is_const(ctx, &safe, &F::ONE);

        make_public.push(result_k);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(pow_elementwise, args);
}
