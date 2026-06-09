// DS-1000 case 374 — Halo2 baseline.
// Kernel: ecdf_result returns (xs, ys). For each `eval[i]`:
//   element < xs[0]            → 0
//   element >= xs[-1]          → 1
//   else                       → ys[(xs > element).argmax() - 1]
// xs is the ascending sort of grades; ys[k] = (k+1)/N.
//
// We unroll the branching as float-arithmetic via select. The
// `(xs > element).argmax()` is the first-occurrence index — implemented
// as a running selector that only commits once.

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
    pub grades: Vec<f64>,    // 50 floats
    pub eval: Vec<String>,   // 5 signed integers
}

fn ecdf_interp<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 50;
    const M: usize = 5;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.grades.len(), N);
    assert_eq!(input.eval.len(), M);

    let grades: Vec<AssignedValue<F>> = input.grades.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // Sort grades → xs ascending (selection sort network).
    let mut xs = grades.clone();
    for i in 0..N {
        for j in (i + 1)..N {
            let lt = rc.is_less_than(ctx, xs[j], xs[i], 128);
            let new_i = gate.select(ctx, xs[j], xs[i], lt);
            let new_j = gate.select(ctx, xs[i], xs[j], lt);
            xs[i] = new_i;
            xs[j] = new_j;
        }
    }

    // ys[k] = (k+1)/N as quantised constants.
    let n_fp = ctx.load_witness(fpc.quantization(N as f64));
    let ys: Vec<AssignedValue<F>> = (0..N).map(|k| {
        let num = ctx.load_witness(fpc.quantization((k + 1) as f64));
        fpc.qdiv(ctx, num, n_fp)
    }).collect();

    let one_fp = ctx.load_witness(fpc.quantization(1.0));
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));

    // Load eval integers, then cast to fp (mul by 1.0).
    let eval_fp: Vec<AssignedValue<F>> = input.eval.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        let signed = if s.starts_with('-') { gate.neg(ctx, abs) } else { abs };
        fpc.qmul(ctx, signed, one_fp)
    }).collect();

    for i in 0..M {
        let e = eval_fp[i];

        // element < xs[0]
        let lt_lo = rc.is_less_than(ctx, e, xs[0], 128);
        // element >= xs[N-1]  ↔  !(element < xs[N-1])
        let lt_hi = rc.is_less_than(ctx, e, xs[N - 1], 128);
        let ge_hi = gate.not(ctx, lt_hi);

        // First-occurrence argmax of (xs > element). best_idx starts at 0,
        // commits on first hit.
        let mut found = Constant(F::ZERO);
        let mut best = ctx.load_constant(F::ZERO);
        for k in 0..N {
            // xs[k] > e  ↔  e < xs[k]
            let hit = rc.is_less_than(ctx, e, xs[k], 128);
            let not_found = gate.not(ctx, found);
            let commit = gate.and(ctx, hit, not_found);
            let k_const = Constant(F::from(k as u64));
            best = gate.select(ctx, k_const, best, commit);
            let found_av = gate.or(ctx, found, hit);
            found = halo2_base::QuantumCell::Existing(found_av);
        }
        // idx = best - 1. Looked up by computing ys[best-1] via a select
        // chain. best is in 1..=N-1 in the else branch (guaranteed by
        // the elif/else).
        let mut interp = ys[0];
        for k in 1..N {
            let is_k = gate.is_equal(ctx, best, Constant(F::from(k as u64)));
            interp = gate.select(ctx, ys[k - 1], interp, is_k);
        }

        // result[i] = lt_lo ? 0 : (ge_hi ? 1 : interp)
        let v_hi = gate.select(ctx, one_fp, interp, ge_hi);
        let v = gate.select(ctx, zero_fp, v_hi, lt_lo);
        make_public.push(v);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(ecdf_interp, args);
}
