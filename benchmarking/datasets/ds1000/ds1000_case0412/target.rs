// DS-1000 case 412 — Halo2 baseline (compact-output version).
// Kernel: `x[x >= 0]` on Float[10].
//
// Output contract mirrors Zinnia's dynamic-array encoding: kept values
// compacted to the front in source order, zero-padded, preceded by the
// kept count exposed twice (runtime length + shape dim). Compaction is
// paid in-circuit via prefix counts + a triangular indicator gather.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub x: Vec<f64> }

fn filter_nonneg_compact<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 10;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.x.len(), N);

    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    let cells: Vec<AssignedValue<F>> = input.x.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // keep[i] = !(x < 0); prefix[i] = #kept before i.
    let zero = ctx.load_constant(F::ZERO);
    let mut keeps = Vec::with_capacity(N);
    let mut count = zero;
    for x in &cells {
        let lt = rc.is_less_than(ctx, *x, zero_fp, 128);
        let keep = gate.not(ctx, lt);
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
            outs[j] = gate.select(ctx, cells[i], outs[j], here);
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
fn main() { env_logger::init(); let args = Cli::parse(); run(filter_nonneg_compact, args); }
