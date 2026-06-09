// DS-1000 case 480 — Halo2 baseline.
// Kernel: result = ((x == a) & (y == b)).argmax();
//         if x[result] != a or y[result] != b: result = -1
// On Integer[8] × Integer[8] + 2 scalar ints.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x: Vec<String>, pub y: Vec<String>, pub a: String, pub b_val: String }

fn pred_argmax<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let ctx = b.main(0);
    assert_eq!(input.x.len(), 8);
    assert_eq!(input.y.len(), 8);
    let load = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(s.trim_start_matches('-').parse::<u128>().unwrap()));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };
    let xs: Vec<AssignedValue<F>> = input.x.iter().map(|s| load(ctx, s)).collect();
    let ys: Vec<AssignedValue<F>> = input.y.iter().map(|s| load(ctx, s)).collect();
    let a = load(ctx, &input.a);
    let b_cell = load(ctx, &input.b_val);

    // Per-index mask = (x[i] == a) & (y[i] == b).
    let mut masks: Vec<AssignedValue<F>> = Vec::with_capacity(8);
    for k in 0..8 {
        let xa = gate.is_equal(ctx, xs[k], a);
        let yb = gate.is_equal(ctx, ys[k], b_cell);
        masks.push(gate.and(ctx, xa, yb));
    }
    // argmax over the bit mask: first index where bit is 1.
    let mut best_val = masks[0];
    let mut best_idx = ctx.load_constant(F::ZERO);
    for k in 1..8usize {
        let is_new = rc.is_less_than(ctx, best_val, masks[k], 128);
        best_val = gate.select(ctx, masks[k], best_val, is_new);
        let idx_k = Constant(F::from(k as u64));
        best_idx = gate.select(ctx, idx_k, best_idx, is_new);
    }
    // If no mask was true (best_val == 0), result = -1.
    let one = ctx.load_constant(F::ONE);
    let minus_one = gate.neg(ctx, one);
    let result = gate.select(ctx, best_idx, minus_one, best_val);
    make_public.push(result);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(pred_argmax, args); }
