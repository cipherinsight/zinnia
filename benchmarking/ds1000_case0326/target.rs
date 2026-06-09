// DS-1000 case 326 — Halo2 baseline.
// Kernel: `np.degrees(np.arcsin(value))`. Mirrors Zinnia IR (11 stmts):
// ReadF + 3× ConstantF + ArcSinFIR + GTE + LTE + And + Assert + MulF + ExposeF.
// Asserts -1 ≤ value ≤ 1 (arcsin domain), then 180/π scale.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub value: f64 }

fn arcsin_degrees<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    let v_fp = ctx.load_witness(fpc.quantization(input.value));
    let one_neg = ctx.load_witness(fpc.quantization(-1.0));
    let one_pos = ctx.load_witness(fpc.quantization(1.0));
    let one80_over_pi = ctx.load_witness(fpc.quantization(180.0 / std::f64::consts::PI));
    // Assert -1 ≤ v ≤ 1: implemented as (v >= -1) AND (v <= 1).
    let v_ge_m1 = {
        let lt = rc.is_less_than(ctx, v_fp, one_neg, 128);
        gate.not(ctx, lt)
    };
    let v_le_1 = {
        let lt = rc.is_less_than(ctx, one_pos, v_fp, 128);
        gate.not(ctx, lt)
    };
    let in_range = gate.and(ctx, v_ge_m1, v_le_1);
    gate.assert_is_const(ctx, &in_range, &F::ONE);
    // halo2-graph has no qarcsin — use the witness-and-verify pattern
    // (same idea as its qsqrt): witness y = arcsin(value), then check
    // sin(y) ≈ value within ±0.001. y is also range-bounded to
    // [-π/2, π/2] to make arcsin unique.
    let y_f64 = input.value.asin();
    let as_val = ctx.load_witness(fpc.quantization(y_f64));
    let sin_y = fpc.qsin(ctx, as_val);
    let diff = fpc.qsub(ctx, sin_y, v_fp);
    let eps_pos = ctx.load_witness(fpc.quantization(0.001));
    let eps_neg = ctx.load_witness(fpc.quantization(-0.001));
    let lt_p = rc.is_less_than(ctx, diff, eps_pos, 128);
    let gt_n = rc.is_less_than(ctx, eps_neg, diff, 128);
    gate.assert_is_const(ctx, &lt_p, &F::ONE);
    gate.assert_is_const(ctx, &gt_n, &F::ONE);
    // Range-check y ∈ [-π/2, π/2].
    let pi_half_p = ctx.load_witness(fpc.quantization(std::f64::consts::FRAC_PI_2));
    let pi_half_n = ctx.load_witness(fpc.quantization(-std::f64::consts::FRAC_PI_2));
    let y_ge_n = { let lt = rc.is_less_than(ctx, as_val, pi_half_n, 128); gate.not(ctx, lt) };
    let y_le_p = { let lt = rc.is_less_than(ctx, pi_half_p, as_val, 128); gate.not(ctx, lt) };
    let y_in = gate.and(ctx, y_ge_n, y_le_p);
    gate.assert_is_const(ctx, &y_in, &F::ONE);

    let degrees = fpc.qmul(ctx, as_val, one80_over_pi);
    make_public.push(degrees);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(arcsin_degrees, args); }
