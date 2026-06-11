// DS-1000 case 416 — Halo2 baseline (fair dynamic version).
// Kernel: data[:, :(M//bs)*bs].reshape(R, -1, bs).mean(axis=-1) on
// Float[8,6] with a RUNTIME bin_size witness.
//
// Honest contract (mirrors the Zinnia lowering):
//   * bin_size is a circuit witness; n_bins = 6 // bin_size is a witness
//     constrained by n_bins*bs + rem == 6, 0 <= rem < bs (64-bit domain).
//   * Circuit shape is FIXED (8x6 output grid, 6x6 window-mask grid);
//     runtime extent is carried by masks, not host-side loop bounds.
//   * Window masks are hoisted: computed once per (slot, col) pair and
//     shared across all 8 rows.
//   * Per-bin mean = floor(sum / bs) on quantized values, enforced as
//     sum == quot*bs + rmd with 0 <= rmd < bs.
//   * Exposes n_bins and 8*n_bins (Zinnia's dyn metadata), then 48
//     masked outputs (row-major, slot j holds the bin-j mean for
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

const REVERSE_COLS: bool = false;

fn bin_mean_2d_dyn<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    const ROWS: usize = 8;
    const COLS: usize = 6;
    let gate = GateChip::<F>::default();
    let rc = b.range_chip();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.data.len(), ROWS * COLS);

    // Quantized data cells (optionally column-reversed at load — a
    // compile-time permutation, no gates).
    let q = |v: f64| -> u128 { (v * (1u128 << P) as f64).round() as u128 };
    let mut vals = vec![0u128; ROWS * COLS];
    let mut cells = Vec::with_capacity(ROWS * COLS);
    for i in 0..ROWS {
        for s in 0..COLS {
            let src = if REVERSE_COLS { COLS - 1 - s } else { s };
            let v = input.data[i * COLS + src];
            vals[i * COLS + s] = q(v);
            cells.push(ctx.load_witness(fpc.quantization(v)));
        }
    }

    // Runtime bin_size + floor-div witnesses, constrained in-circuit.
    let bs_u = input.bin_size as u64;
    let bs = ctx.load_witness(F::from(bs_u));
    let n_bins_u = (COLS as u64) / bs_u.max(1);
    let rem_u = (COLS as u64) % bs_u.max(1);
    let n_bins = ctx.load_witness(F::from(n_bins_u));
    let rem = ctx.load_witness(F::from(rem_u));
    rc.range_check(ctx, bs, 64);
    rc.range_check(ctx, n_bins, 64);
    rc.range_check(ctx, rem, 64);
    let prod = gate.mul(ctx, n_bins, bs);
    let total = gate.add(ctx, prod, rem);
    let cols_c = ctx.load_constant(F::from(COLS as u64));
    let ok = gate.is_equal(ctx, total, cols_c);
    gate.assert_is_const(ctx, &ok, &F::ONE);
    rc.check_less_than(ctx, rem, bs, 64);

    // Metadata (Zinnia's dyn-array encoding: last dim + runtime length).
    let rows_c = ctx.load_constant(F::from(ROWS as u64));
    let length = gate.mul(ctx, n_bins, rows_c);
    make_public.push(n_bins);
    make_public.push(length);

    // Hoisted window masks: m[j][s] = (j*bs <= s < j*bs + bs), and the
    // per-slot activity j < n_bins. Shared by every row.
    let zero = ctx.load_constant(F::ZERO);
    let mut masks = vec![vec![zero; COLS]; COLS];
    let mut actives = vec![zero; COLS];
    for j in 0..COLS {
        let j_c = ctx.load_constant(F::from(j as u64));
        let lo = gate.mul(ctx, j_c, bs);
        let hi = gate.add(ctx, lo, bs);
        actives[j] = rc.is_less_than(ctx, j_c, n_bins, 64);
        for s in 0..COLS {
            let s_c = ctx.load_constant(F::from(s as u64));
            let lt_lo = rc.is_less_than(ctx, s_c, lo, 64);
            let ge_lo = gate.not(ctx, lt_lo);
            let lt_hi = rc.is_less_than(ctx, s_c, hi, 64);
            masks[j][s] = gate.and(ctx, ge_lo, lt_hi);
        }
    }

    // Per (row, slot): masked window sum, then mean = floor(sum / bs)
    // (sum == quot*bs + rmd, 0 <= rmd < bs), masked by slot activity.
    for i in 0..ROWS {
        for j in 0..COLS {
            let mut acc = zero;
            let mut sum_u: u128 = 0;
            for s in 0..COLS {
                let term = gate.mul(ctx, masks[j][s], cells[i * COLS + s]);
                acc = gate.add(ctx, acc, term);
                if (j as u64) < n_bins_u
                    && (j as u64) * bs_u <= s as u64
                    && (s as u64) < (j as u64) * bs_u + bs_u
                {
                    sum_u += vals[i * COLS + s];
                }
            }
            let q_u = sum_u / bs_u.max(1) as u128;
            let r_u = sum_u % bs_u.max(1) as u128;
            let quot = ctx.load_witness(F::from_u128(q_u));
            let rmd = ctx.load_witness(F::from_u128(r_u));
            rc.range_check(ctx, quot, 64);
            rc.range_check(ctx, rmd, 64);
            let qb = gate.mul(ctx, quot, bs);
            let recon = gate.add(ctx, qb, rmd);
            // Inactive slots have masked sum 0 and quot = rmd = 0, so the
            // constraint holds uniformly.
            let eq = gate.is_equal(ctx, recon, acc);
            gate.assert_is_const(ctx, &eq, &F::ONE);
            rc.check_less_than(ctx, rmd, bs, 64);
            let out = gate.select(ctx, quot, zero, actives[j]);
            make_public.push(out);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(bin_mean_2d_dyn, args);
}
