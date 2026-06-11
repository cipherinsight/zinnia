// DS-1000 case 415 — Halo2 baseline (fair dynamic version).
// Kernel: data[:(N//bs)*bs].reshape(-1, bs).max(axis=1) on Float[8] with a
// RUNTIME bin_size witness.
//
// Honest contract (mirrors the Zinnia lowering):
//   * bin_size is a circuit witness, range-checked; n_bins = 8 // bin_size
//     is a witness constrained by n_bins*bs + rem == 8, 0 <= rem < bs.
//   * The circuit shape is FIXED (8 output slots, 8x8 mask grid); the
//     runtime extent is carried by masks, not by host-side loop bounds.
//   * Exposes n_bins twice (runtime length + shape dim, as Zinnia does)
//     followed by 8 masked outputs (slot j holds the bin-j max for
//     j < n_bins, else 0).
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
pub struct CircuitInput { pub data: Vec<f64>, pub bin_size: u32 }

fn bin_max_1d_dyn<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const N: usize = 8;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.data.len(), N);

    let cells: Vec<AssignedValue<F>> = input.data.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();

    // Runtime bin_size + floor-div witnesses, constrained in-circuit
    // (mirrors Zinnia's floor_div discharge): 8 = n_bins*bs + rem,
    // 0 <= rem < bs, all operands in the enforced 64-bit domain.
    let bs_u = input.bin_size as u64;
    let bs = ctx.load_witness(F::from(bs_u));
    let n_bins_u = (N as u64) / bs_u.max(1);
    let rem_u = (N as u64) % bs_u.max(1);
    let n_bins = ctx.load_witness(F::from(n_bins_u));
    let rem = ctx.load_witness(F::from(rem_u));
    rc.range_check(ctx, bs, 64);
    rc.range_check(ctx, n_bins, 64);
    rc.range_check(ctx, rem, 64);
    let prod = gate.mul(ctx, n_bins, bs);
    let total = gate.add(ctx, prod, rem);
    let eight = ctx.load_constant(F::from(N as u64));
    let ok = gate.is_equal(ctx, total, eight);
    gate.assert_is_const(ctx, &ok, &F::ONE);
    rc.check_less_than(ctx, rem, bs, 64);

    // Expose runtime length + shape dim (Zinnia's dyn-array metadata).
    make_public.push(n_bins);
    make_public.push(n_bins);

    let zero = ctx.load_constant(F::ZERO);
    for j in 0..N {
        // Bin-j window: lo = j*bs, hi = lo + bs. Slot active iff j < n_bins.
        let j_c = ctx.load_constant(F::from(j as u64));
        let lo = gate.mul(ctx, j_c, bs);
        let hi = gate.add(ctx, lo, bs);
        let j_active = rc.is_less_than(ctx, j_c, n_bins, 64);

        // Running max over the masked window (seen-bit chain — no
        // sentinel assumptions about the float domain).
        let mut acc = zero;
        let mut seen = zero;
        for (s, cell) in cells.iter().enumerate() {
            let s_c = ctx.load_constant(F::from(s as u64));
            let ge_lo = {
                let lt = rc.is_less_than(ctx, s_c, lo, 64);
                gate.not(ctx, lt)
            };
            let lt_hi = rc.is_less_than(ctx, s_c, hi, 64);
            let in_bin = gate.and(ctx, ge_lo, lt_hi);
            let m = gate.and(ctx, in_bin, j_active);
            let better = rc.is_less_than(ctx, acc, *cell, 128);
            let not_seen = gate.not(ctx, seen);
            let better_or_first = gate.or(ctx, not_seen, better);
            let take = gate.and(ctx, m, better_or_first);
            acc = gate.select(ctx, *cell, acc, take);
            seen = gate.or(ctx, seen, m);
        }
        let out = gate.select(ctx, acc, zero, j_active);
        make_public.push(out);
    }
}

fn main() { env_logger::init(); let args = Cli::parse(); run(bin_max_1d_dyn, args); }
