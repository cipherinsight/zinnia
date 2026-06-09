// DS-1000 case 454 — Halo2 baseline.
// Kernel: `X / np.abs(X).max(axis=1).reshape(-1, 1)` on Integer[5, 4].
// Linf row-normalisation. Mirrors Zinnia IR: 20× ReadInt + 20× FloatCast
// + 20× AbsF + 15× GT + 15× SelectF (row max) + 5× NotEqualF + 20× DivF
// + 20× AssertIR + 20× ExposePublicF.

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
pub struct CircuitInput { pub x: Vec<String> }

fn linf_norm_circuit<F: ScalarField>(
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
    let zero_fp = ctx.load_witness(fixed_point_chip.quantization(0.0));
    let eps_pos = Constant(fixed_point_chip.quantization(0.001));
    let eps_neg = Constant(fixed_point_chip.quantization(-0.001));
    let mut x_fp: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    let mut abs_fp: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    for s in &input.x {
        let v: i64 = s.parse().expect("decimal int");
        let v_fp = ctx.load_witness(fixed_point_chip.quantization(v as f64));
        let lt_zero = range_chip.is_less_than(ctx, v_fp, zero_fp, 128);
        let neg_fp = gate.neg(ctx, v_fp);
        let a_fp = gate.select(ctx, neg_fp, v_fp, lt_zero);
        x_fp.push(v_fp);
        abs_fp.push(a_fp);
    }
    for i in 0..5 {
        let base = i * 4;
        let mut row_max = abs_fp[base];
        for j in 1..4 {
            let cur = abs_fp[base + j];
            let cur_gt = range_chip.is_less_than(ctx, row_max, cur, 128);
            row_max = gate.select(ctx, cur, row_max, cur_gt);
        }
        let diff = fixed_point_chip.qsub(ctx, row_max, zero_fp);
        let lt_p = range_chip.is_less_than(ctx, diff, eps_pos, 128);
        let gt_n = range_chip.is_less_than(ctx, eps_neg, diff, 128);
        let eq_zero = gate.and(ctx, lt_p, gt_n);
        let ne_zero = gate.not(ctx, eq_zero);
        for j in 0..4 {
            gate.assert_is_const(ctx, &ne_zero, &F::ONE);
            let r = fixed_point_chip.qdiv(ctx, x_fp[base + j], row_max);
            make_public.push(r);
        }
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(linf_norm_circuit, args); }
