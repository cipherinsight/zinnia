// DS-1000 case 491 — Halo2 baseline.
// Kernel: `(~np.isclose(s1, s2)).sum()` on Float[6, 1] × Float[6, 1].
// np.isclose default: atol = 1e-8, rtol = 1e-5.
// close[k] = |s1[k] - s2[k]| <= atol + rtol * |s2[k]|
// result = sum_k (1 - close[k]).
//
// Float "close" check via EqualFIR pattern:
//   diff = s1 - s2; eps = atol + rtol*|s2|;
//   close = (-eps < diff) AND (diff < eps).

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
pub struct CircuitInput { pub s1: Vec<f64>, pub s2: Vec<f64> }  // 6 + 6 floats

fn isclose_sum<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 6;
    const ATOL: f64 = 1.0e-8;
    const RTOL: f64 = 1.0e-5;

    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);

    assert_eq!(input.s1.len(), N);
    assert_eq!(input.s2.len(), N);

    let s1: Vec<AssignedValue<F>> = input.s1.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let s2: Vec<AssignedValue<F>> = input.s2.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    let atol = ctx.load_witness(fpc.quantization(ATOL));
    let rtol = ctx.load_witness(fpc.quantization(RTOL));
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));

    let mut count: Option<AssignedValue<F>> = None;
    for k in 0..N {
        // |s2[k]| via sign select.
        let s2_lt0 = rc.is_less_than(ctx, s2[k], zero_fp, 128);
        let neg_s2 = fpc.qsub(ctx, zero_fp, s2[k]);
        let abs_s2 = gate.select(ctx, neg_s2, s2[k], s2_lt0);
        // eps = atol + rtol * |s2[k]|
        let rtol_abs = fpc.qmul(ctx, rtol, abs_s2);
        let eps = fpc.qadd(ctx, atol, rtol_abs);
        let neg_eps = fpc.qsub(ctx, zero_fp, eps);

        // diff = s1 - s2; close = (diff < eps) AND (-eps < diff)
        let diff = fpc.qsub(ctx, s1[k], s2[k]);
        let lt_p = rc.is_less_than(ctx, diff, eps, 128);
        let gt_n = rc.is_less_than(ctx, neg_eps, diff, 128);
        let close = gate.and(ctx, lt_p, gt_n);
        // not_close = 1 - close
        let not_close = gate.not(ctx, close);
        count = Some(match count {
            None => not_close,
            Some(prev) => gate.add(ctx, prev, not_close),
        });
    }
    make_public.push(count.unwrap());
}

fn main() { env_logger::init(); let args = Cli::parse(); run(isclose_sum, args); }
