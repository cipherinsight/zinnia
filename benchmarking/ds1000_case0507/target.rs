// DS-1000 case 507 — Halo2 baseline.
// Kernel:
//     mask = im == 0
//     rows = np.flatnonzero((~mask).sum(axis=1))
//     cols = np.flatnonzero((~mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
// im: Integer[5, 6]. Bounding box of nonzero entries.
//
// Output shape is data-dependent — follow 0303 reshape+trim style:
// expose all 30 cells masked-to-zero outside the bbox, plus shape metadata
// (rmin, rmax, cmin, cmax, height, width).

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub im: Vec<String>, // 5*6 = 30 decimal integers
}

fn nonzero_bbox_5x6<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const R: usize = 5;
    const C: usize = 6;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let ctx = builder.main(0);

    assert_eq!(input.im.len(), R * C);
    let cells: Vec<AssignedValue<F>> = input.im.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int")));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    }).collect();

    // nonzero[i][j] = 1 - (cell == 0).
    let zero = ctx.load_constant(F::ZERO);
    let mut nonzero = Vec::with_capacity(R * C);
    for k in 0..(R * C) {
        let is_zero = gate.is_equal(ctx, cells[k], zero);
        nonzero.push(gate.not(ctx, is_zero));
    }

    // Row-has-nonzero: sum nonzero indicators per row; row_has = (sum != 0).
    let mut row_has = Vec::with_capacity(R);
    for i in 0..R {
        let mut s = nonzero[i * C];
        for j in 1..C { s = gate.add(ctx, s, nonzero[i * C + j]); }
        let is_z = gate.is_equal(ctx, s, zero);
        row_has.push(gate.not(ctx, is_z));
    }
    let mut col_has = Vec::with_capacity(C);
    for j in 0..C {
        let mut s = nonzero[j];
        for i in 1..R { s = gate.add(ctx, s, nonzero[i * C + j]); }
        let is_z = gate.is_equal(ctx, s, zero);
        col_has.push(gate.not(ctx, is_z));
    }

    // rmin = smallest i with row_has[i]; sentinel = R-1 (numpy raises on empty).
    let mut rmin = ctx.load_constant(F::from((R - 1) as u64));
    let mut found = ctx.load_constant(F::ZERO);
    for i in 0..R {
        let i_const = Constant(F::from(i as u64));
        let not_found = gate.not(ctx, found);
        let take = gate.mul(ctx, not_found, row_has[i]);
        rmin = gate.select(ctx, i_const, rmin, take);
        let or_v = gate.add(ctx, found, row_has[i]);
        let and_v = gate.mul(ctx, found, row_has[i]);
        found = gate.sub(ctx, or_v, and_v);
    }
    // rmax = largest i with row_has[i]: monotonic select.
    let mut rmax = ctx.load_constant(F::ZERO);
    for i in 0..R {
        let i_const = Constant(F::from(i as u64));
        rmax = gate.select(ctx, i_const, rmax, row_has[i]);
    }
    let mut cmin = ctx.load_constant(F::from((C - 1) as u64));
    let mut cfound = ctx.load_constant(F::ZERO);
    for j in 0..C {
        let j_const = Constant(F::from(j as u64));
        let not_found = gate.not(ctx, cfound);
        let take = gate.mul(ctx, not_found, col_has[j]);
        cmin = gate.select(ctx, j_const, cmin, take);
        let or_v = gate.add(ctx, cfound, col_has[j]);
        let and_v = gate.mul(ctx, cfound, col_has[j]);
        cfound = gate.sub(ctx, or_v, and_v);
    }
    let mut cmax = ctx.load_constant(F::ZERO);
    for j in 0..C {
        let j_const = Constant(F::from(j as u64));
        cmax = gate.select(ctx, j_const, cmax, col_has[j]);
    }

    // Output: cells masked to zero outside bbox [rmin..=rmax] × [cmin..=cmax].
    let one = ctx.load_constant(F::ONE);
    let rmax_p1 = gate.add(ctx, rmax, one);
    let cmax_p1 = gate.add(ctx, cmax, one);
    for i in 0..R {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let i_p1 = gate.add(ctx, i_cell, one);
        let ge_rmin = rc.is_less_than(ctx, rmin, i_p1, 32);
        let lt_rmax_p1 = rc.is_less_than(ctx, i_cell, rmax_p1, 32);
        let in_row = gate.and(ctx, ge_rmin, lt_rmax_p1);
        for j in 0..C {
            let j_cell = ctx.load_constant(F::from(j as u64));
            let j_p1 = gate.add(ctx, j_cell, one);
            let ge_cmin = rc.is_less_than(ctx, cmin, j_p1, 32);
            let lt_cmax_p1 = rc.is_less_than(ctx, j_cell, cmax_p1, 32);
            let in_col = gate.and(ctx, ge_cmin, lt_cmax_p1);
            let in_box = gate.and(ctx, in_row, in_col);
            let out = gate.select(ctx, cells[i * C + j], zero, in_box);
            make_public.push(out);
        }
    }

    // Shape metadata.
    let height = gate.sub(ctx, rmax_p1, rmin);
    let width = gate.sub(ctx, cmax_p1, cmin);
    make_public.push(rmin);
    make_public.push(rmax);
    make_public.push(cmin);
    make_public.push(cmax);
    make_public.push(height);
    make_public.push(width);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(nonzero_bbox_5x6, args);
}
