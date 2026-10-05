// DS-1000 case 431 — Halo2 baseline.
// Kernel: `~np.logical_and(a > mean-2σ, a < mean+2σ)` on Integer[8].
// Outlier mask: True where the value is outside the ±2σ interval.
// Mirrors Zinnia IR: 8× ReadInt + 12× FloatCast + sum/div + ²/√/div for σ
// + 8× GT + 8× LT + 8× LogicalAnd + 8× LogicalNot + 8× ExposePublicII.

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
pub struct CircuitInput { pub a: Vec<String> }  // 8 decimal integers

fn outlier_mask<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 8);
    // Load int inputs, cast each to fp (matches Zinnia's FloatCastIR).
    let cells_fp: Vec<AssignedValue<F>> = input.a.iter().map(|s| {
        let v: i64 = s.parse().expect("decimal int");
        ctx.load_witness(fixed_point_chip.quantization(v as f64))
    }).collect();
    // mean
    let mut sum = cells_fp[0];
    for k in 1..8 { sum = fixed_point_chip.qadd(ctx, sum, cells_fp[k]); }
    let n = ctx.load_witness(fixed_point_chip.quantization(8.0));
    let mean = fixed_point_chip.qdiv(ctx, sum, n);
    // variance / sqrt — witness-and-verify pattern
    let mut sq_sum: Option<AssignedValue<F>> = None;
    for c in &cells_fp {
        let d = fixed_point_chip.qsub(ctx, *c, mean);
        let d2 = fixed_point_chip.qmul(ctx, d, d);
        sq_sum = Some(match sq_sum { None => d2, Some(p) => fixed_point_chip.qadd(ctx, p, d2) });
    }
    let variance = fixed_point_chip.qdiv(ctx, sq_sum.unwrap(), n);
    let v_f64 = fixed_point_chip.dequantization(*variance.value());
    let s_f64 = if v_f64 < 0.0 { 0.0 } else { v_f64.sqrt() };
    let std_cell = ctx.load_witness(fixed_point_chip.quantization(s_f64));
    let s2 = fixed_point_chip.qmul(ctx, std_cell, std_cell);
    let diff_sq = fixed_point_chip.qsub(ctx, s2, variance);
    let eps_pos = Constant(fixed_point_chip.quantization(0.001));
    let eps_neg = Constant(fixed_point_chip.quantization(-0.001));
    let sq_lt = range_chip.is_less_than(ctx, diff_sq, eps_pos, 128);
    let sq_gt = range_chip.is_less_than(ctx, eps_neg, diff_sq, 128);
    gate.assert_is_const(ctx, &sq_lt, &F::ONE);
    gate.assert_is_const(ctx, &sq_gt, &F::ONE);

    let two = ctx.load_witness(fixed_point_chip.quantization(2.0));
    let two_std = fixed_point_chip.qmul(ctx, two, std_cell);
    let lo = fixed_point_chip.qsub(ctx, mean, two_std);
    let hi = fixed_point_chip.qadd(ctx, mean, two_std);
    // For each cell: outlier = NOT (a > lo AND a < hi)
    for c in &cells_fp {
        let gt_lo = range_chip.is_less_than(ctx, lo, *c, 128);
        let lt_hi = range_chip.is_less_than(ctx, *c, hi, 128);
        let inside = gate.and(ctx, gt_lo, lt_hi);
        let outside = gate.not(ctx, inside);
        make_public.push(outside);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(outlier_mask, args); }
