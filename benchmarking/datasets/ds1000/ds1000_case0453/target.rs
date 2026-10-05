// DS-1000 case 453 — Halo2 baseline.
// Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1, 1)` on Integer[5, 4].
// L2 row-normalisation; result is Float[5, 4].
// Mirrors Zinnia IR (164 stmts): 20× ReadInt + 20× FloatCast + 20× MulI (squared)
// + 15× AddI (row sums) + 5× Sqrt + 5× NotEqualF + 20× DivF + 25× AssertIR
// + 20× ExposePublicF.

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
pub struct CircuitInput { pub x: Vec<String> }  // 20 decimal integers

fn l2_norm_circuit<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.x.len(), 20);
    let x_fp: Vec<AssignedValue<F>> = input.x.iter().map(|s| {
        let v: i64 = s.parse().expect("decimal int");
        ctx.load_witness(fixed_point_chip.quantization(v as f64))
    }).collect();
    let zero_fp = ctx.load_witness(fixed_point_chip.quantization(0.0));
    let eps_pos = Constant(fixed_point_chip.quantization(0.001));
    let eps_neg = Constant(fixed_point_chip.quantization(-0.001));
    for i in 0..5 {
        let base = i * 4;
        // sum of squares
        let mut sq_sum: Option<AssignedValue<F>> = None;
        for j in 0..4 {
            let sq = fixed_point_chip.qmul(ctx, x_fp[base + j], x_fp[base + j]);
            sq_sum = Some(match sq_sum { None => sq, Some(p) => fixed_point_chip.qadd(ctx, p, sq) });
        }
        let ss = sq_sum.unwrap();
        // witness-and-verify sqrt
        let ss_f64 = fixed_point_chip.dequantization(*ss.value());
        let l2_f64 = if ss_f64 < 0.0 { 0.0 } else { ss_f64.sqrt() };
        let l2 = ctx.load_witness(fixed_point_chip.quantization(l2_f64));
        let l2_sq = fixed_point_chip.qmul(ctx, l2, l2);
        let diff_sq = fixed_point_chip.qsub(ctx, l2_sq, ss);
        let sq_lt = range_chip.is_less_than(ctx, diff_sq, eps_pos, 128);
        let sq_gt = range_chip.is_less_than(ctx, eps_neg, diff_sq, 128);
        gate.assert_is_const(ctx, &sq_lt, &F::ONE);
        gate.assert_is_const(ctx, &sq_gt, &F::ONE);
        // l2 != 0 safety
        let diff = fixed_point_chip.qsub(ctx, l2, zero_fp);
        let lt_p = range_chip.is_less_than(ctx, diff, eps_pos, 128);
        let gt_n = range_chip.is_less_than(ctx, eps_neg, diff, 128);
        let eq_zero = gate.and(ctx, lt_p, gt_n);
        let ne_zero = gate.not(ctx, eq_zero);
        for j in 0..4 {
            gate.assert_is_const(ctx, &ne_zero, &F::ONE);
            let r = fixed_point_chip.qdiv(ctx, x_fp[base + j], l2);
            make_public.push(r);
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(l2_norm_circuit, args); }
