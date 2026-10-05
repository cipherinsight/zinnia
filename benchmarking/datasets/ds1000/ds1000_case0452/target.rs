// DS-1000 case 452 — Halo2 baseline.
// Kernel: `X / np.abs(X).sum(axis=1).reshape(-1, 1)` on Integer[5, 4]
// (L1 row-normalisation; result is Float[5, 4]).
// Mirrors Zinnia IR (142 stmts): 20× ReadInt + 20× FloatCast + 20× AbsF
// + 15× AddF (row sums) + 5× NotEqualF (l1 ≠ 0 guard) + 20× DivF
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
pub struct CircuitInput {
    pub x: Vec<String>,  // 20 decimal integers, row-major (5 rows × 4 cols)
}

fn l1_norm_circuit<F: ScalarField>(
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

    // Load each integer input directly as a fp-quantised float (matches
    // Zinnia's ReadInt + FloatCast collapsing into a witnessed fp cell).
    let mut x_fp: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    let mut abs_fp: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    let zero_fp = ctx.load_witness(fixed_point_chip.quantization(0.0));
    let eps_pos = Constant(fixed_point_chip.quantization(0.001));
    let eps_neg = Constant(fixed_point_chip.quantization(-0.001));

    for s in &input.x {
        let v: i64 = s.parse().expect("decimal int");
        let v_fp = ctx.load_witness(fixed_point_chip.quantization(v as f64));
        // abs via sign select on the fp value.
        let lt_zero = range_chip.is_less_than(ctx, v_fp, zero_fp, 128);
        let neg_fp = gate.neg(ctx, v_fp);
        let a_fp = gate.select(ctx, neg_fp, v_fp, lt_zero);
        x_fp.push(v_fp);
        abs_fp.push(a_fp);
    }

    for i in 0..5 {
        let base = i * 4;
        let s01 = fixed_point_chip.qadd(ctx, abs_fp[base], abs_fp[base + 1]);
        let s012 = fixed_point_chip.qadd(ctx, s01, abs_fp[base + 2]);
        let l1 = fixed_point_chip.qadd(ctx, s012, abs_fp[base + 3]);

        // Divide-by-zero safety: assert l1 != 0 (NotEqualFIR + AssertIR per cell).
        let diff = fixed_point_chip.qsub(ctx, l1, zero_fp);
        let lt_p = range_chip.is_less_than(ctx, diff, eps_pos, 128);
        let gt_n = range_chip.is_less_than(ctx, eps_neg, diff, 128);
        let eq_zero = gate.and(ctx, lt_p, gt_n);
        let ne_zero = gate.not(ctx, eq_zero);
        for j in 0..4 {
            gate.assert_is_const(ctx, &ne_zero, &F::ONE);
            let r = fixed_point_chip.qdiv(ctx, x_fp[base + j], l1);
            make_public.push(r);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(l1_norm_circuit, args);
}
