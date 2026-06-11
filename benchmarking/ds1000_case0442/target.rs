// DS-1000 case 442 — Halo2 baseline (compact-output version).
// Kernel: C = A[~np.in1d(A, B)] on Integer[20] × Integer[3].
//
// Output contract mirrors Zinnia's dynamic-array encoding: the kept
// elements compacted to the front in source order, zero-padded to the
// static bound, preceded by the kept count exposed twice (runtime length
// + shape dim). Compaction is paid in-circuit: prefix counts route each
// kept element to its output slot via a triangular indicator gather
// (out[k] = sum_{i>=k} (keep_i && prefix_i == k) * A[i]).

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<i64>, pub b: Vec<i64> }

fn in1d_neg_compact<F: ScalarField>(
    bld: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 20;
    let gate = GateChip::<F>::default();
    // FixedPointChip is constructed only to align with the project's harness; not used.
    let _fpc = FixedPointChip::<F, P>::default(bld);
    let ctx = bld.main(0);
    assert_eq!(input.a.len(), N);
    assert_eq!(input.b.len(), 3);

    let a_cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(F::from(*v as u64))).collect();
    let b_cells: Vec<AssignedValue<F>> = input.b.iter()
        .map(|v| ctx.load_witness(F::from(*v as u64))).collect();

    // keep[i] = !any_j(A[i] == B[j]); prefix[i] = #kept before i.
    let zero = ctx.load_constant(F::ZERO);
    let mut keeps = Vec::with_capacity(N);
    let mut count = zero;
    for i in 0..N {
        let e0 = gate.is_equal(ctx, a_cells[i], b_cells[0]);
        let e1 = gate.is_equal(ctx, a_cells[i], b_cells[1]);
        let e2 = gate.is_equal(ctx, a_cells[i], b_cells[2]);
        let or01 = gate.or(ctx, e0, e1);
        let any_eq = gate.or(ctx, or01, e2);
        let keep = gate.not(ctx, any_eq);
        keeps.push(keep);
        count = gate.add(ctx, count, keep);
    }

    // Runtime length + shape dim (Zinnia's dyn-array metadata).
    make_public.push(count);
    make_public.push(count);

    // Compact gather via one-hot shift network (cheaper than per-slot
    // prefix-equality tests): R is the one-hot of the next free output
    // slot. For each element: deposit into the hot slot when kept, then
    // shift R right by one iff kept. Triangular: after i elements at
    // most slots 0..=i can be hot.
    let mut outs = vec![zero; N];
    let one = ctx.load_constant(F::ONE);
    let mut hot = vec![zero; N];
    hot[0] = one;
    for i in 0..N {
        let lim = (i + 1).min(N);
        for j in 0..lim {
            let here = gate.and(ctx, hot[j], keeps[i]);
            outs[j] = gate.select(ctx, a_cells[i], outs[j], here);
        }
        // Shift the one-hot right by one iff kept.
        let prev: Vec<_> = hot.clone();
        let lim2 = (i + 2).min(N);
        for j in (0..lim2).rev() {
            let shifted = if j == 0 { zero } else { prev[j - 1] };
            hot[j] = gate.select(ctx, shifted, prev[j], keeps[i]);
        }
    }
    for o in outs {
        make_public.push(o);
    }
}
fn main() { env_logger::init(); let args = Cli::parse(); run(in1d_neg_compact, args); }
