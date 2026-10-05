// DS-1000 case 444 — Halo2 baseline.
// Kernel: `C = A[(A > B[0] & A < B[1]) | (A > B[1] & A < B[2])]` on
// Integer[20] × Integer[3]. Dynamic-length output; halo2 unrolls to N = 20
// (padded buffer + length).
//
// Per i: keep_i = (a[i] > B[0] AND a[i] < B[1]) OR (a[i] > B[1] AND a[i] < B[2]).
// Compaction uses the running rank trick (same shape as case 443).

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,
    pub b: Vec<String>,
    pub claimed_len: String,
    pub claimed: Vec<String>,
}

fn range_union<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 20);
    assert_eq!(input.b.len(), 3);
    assert_eq!(input.claimed.len(), 20);

    let load_int = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int"),
        ));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    };

    let a: Vec<AssignedValue<F>> = input.a.iter().map(|s| load_int(ctx, s)).collect();
    let b: Vec<AssignedValue<F>> = input.b.iter().map(|s| load_int(ctx, s)).collect();
    let claimed_len = load_int(ctx, &input.claimed_len);
    let claimed: Vec<AssignedValue<F>> = input.claimed.iter().map(|s| load_int(ctx, s)).collect();

    // keep_i = (a > b[0] AND a < b[1]) OR (a > b[1] AND a < b[2]).
    let mut keep: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    for i in 0..20 {
        let gt_b0 = range_chip.is_less_than(ctx, b[0], a[i], 128);
        let lt_b1 = range_chip.is_less_than(ctx, a[i], b[1], 128);
        let gt_b1 = range_chip.is_less_than(ctx, b[1], a[i], 128);
        let lt_b2 = range_chip.is_less_than(ctx, a[i], b[2], 128);
        let band1 = gate.and(ctx, gt_b0, lt_b1);
        let band2 = gate.and(ctx, gt_b1, lt_b2);
        let m = gate.or(ctx, band1, band2);
        keep.push(m);
    }

    let zero = ctx.load_constant(F::ZERO);
    let mut rank_after: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    let mut running = zero;
    for i in 0..20 {
        running = gate.add(ctx, running, keep[i]);
        rank_after.push(running);
    }
    let total = rank_after[19];

    let mut out: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    for k in 0..20 {
        let target = ctx.load_constant(F::from((k + 1) as u64));
        let mut chosen = zero;
        for i in 0..20 {
            let rank_eq = gate.is_equal(ctx, rank_after[i], target);
            let pick = gate.and(ctx, keep[i], rank_eq);
            chosen = gate.select(ctx, a[i], chosen, pick);
        }
        out.push(chosen);
    }

    let len_ok = gate.is_equal(ctx, total, claimed_len);
    gate.assert_is_const(ctx, &len_ok, &F::ONE);
    for k in 0..20 {
        let eq = gate.is_equal(ctx, out[k], claimed[k]);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }

    make_public.push(total);
    for v in &out { make_public.push(*v); }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(range_union, args); }
