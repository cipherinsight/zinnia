// DS-1000 case 443 — Halo2 baseline.
// Kernel: `C = A[np.in1d(A, B)]` on Integer[20] × Integer[3].
// Returns elements of A that appear in B.
//
// Dynamic output length: the output is unrolled to the static upper bound
// (N = 20) as a padded buffer (kept values, then 0 default). The actual
// keep-count is exposed alongside the padded buffer.
//
// We compute, for each i in 0..20:
//   keep_i = OR over j in 0..3 of (a[i] == b[j])
// Then the output buffer is a compaction: out[k] = a[idx(k)] where idx(k) is
// the k-th i with keep_i = 1. We realise the compaction by a running-index
// gather: for k in 0..20, out[k] = sum of (rank_i == k) * a[i] (with
// rank_i = prefix sum of keep_i). The selection at each (k, i) is an
// equality check on the running rank.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<String>,        // 20 decimal integers
    pub b: Vec<String>,        // 3 decimal integers
    pub claimed_len: String,   // actual output length (<= 20)
    pub claimed: Vec<String>,  // 20 decimal integers (padded with 0)
}

fn in1d_filter<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    let gate = GateChip::<F>::default();
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

    // keep_i = OR_{j} (a[i] == b[j]).
    let mut keep: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    for i in 0..20 {
        let mut acc: Option<AssignedValue<F>> = None;
        for j in 0..3 {
            let eq = gate.is_equal(ctx, a[i], b[j]);
            acc = Some(match acc {
                None => eq,
                Some(p) => gate.or(ctx, p, eq),
            });
        }
        keep.push(acc.unwrap());
    }

    // Running rank: rank_after_i = sum_{i' <= i} keep_{i'}.
    // For the k-th output, we pick a[i] iff keep_i == 1 AND rank_after_i == k + 1.
    let zero = ctx.load_constant(F::ZERO);
    let mut out: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    let mut rank_after: Vec<AssignedValue<F>> = Vec::with_capacity(20);
    let mut running = zero;
    for i in 0..20 {
        running = gate.add(ctx, running, keep[i]);
        rank_after.push(running);
    }
    // Total kept count.
    let total = rank_after[19];

    for k in 0..20 {
        let target = ctx.load_constant(F::from((k + 1) as u64));
        let mut chosen = zero;
        for i in 0..20 {
            let rank_eq = gate.is_equal(ctx, rank_after[i], target);
            let pick = gate.and(ctx, keep[i], rank_eq);
            chosen = gate.select(ctx, a[i], chosen, pick);
        }
        // For positions beyond the kept count, chosen stays at 0 (default).
        out.push(chosen);
    }

    // Assert total == claimed_len and out == claimed (full padded buffer).
    let len_ok = gate.is_equal(ctx, total, claimed_len);
    gate.assert_is_const(ctx, &len_ok, &F::ONE);
    for k in 0..20 {
        let eq = gate.is_equal(ctx, out[k], claimed[k]);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }

    make_public.push(total);
    for v in &out { make_public.push(*v); }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(in1d_filter, args); }
