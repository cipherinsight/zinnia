// DS-1000 case 508 — Halo2 baseline.
// Kernel:
//     B = np.argwhere(A)
//     (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
//     result = A[ystart:ystop, xstart:xstop]
// A: Integer[10, 10]. Same bounding-box-of-nonzero structure as 0507, just
// expressed via np.argwhere + min/max.
//
// Output shape is data-dependent — expose all 100 cells masked-to-zero
// outside the bbox plus shape metadata (ystart, ystop, xstart, xstop, h, w).

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
    pub a: Vec<String>, // 10*10 = 100 decimal integers
}

fn argwhere_bbox_10x10<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const N: usize = 10;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), N * N);
    let cells: Vec<AssignedValue<F>> = input.a.iter().map(|s| {
        let abs = ctx.load_witness(F::from_u128(
            s.trim_start_matches('-').parse::<u128>().expect("decimal int")));
        if s.starts_with('-') { gate.neg(ctx, abs) } else { abs }
    }).collect();

    let zero = ctx.load_constant(F::ZERO);
    let mut nonzero = Vec::with_capacity(N * N);
    for k in 0..(N * N) {
        let is_zero = gate.is_equal(ctx, cells[k], zero);
        nonzero.push(gate.not(ctx, is_zero));
    }

    // row_has[i] = OR_j nonzero[i][j]; col_has[j] = OR_i nonzero[i][j].
    let mut row_has = Vec::with_capacity(N);
    for i in 0..N {
        let mut s = nonzero[i * N];
        for j in 1..N { s = gate.add(ctx, s, nonzero[i * N + j]); }
        let is_z = gate.is_equal(ctx, s, zero);
        row_has.push(gate.not(ctx, is_z));
    }
    let mut col_has = Vec::with_capacity(N);
    for j in 0..N {
        let mut s = nonzero[j];
        for i in 1..N { s = gate.add(ctx, s, nonzero[i * N + j]); }
        let is_z = gate.is_equal(ctx, s, zero);
        col_has.push(gate.not(ctx, is_z));
    }

    // ystart = first i with row_has[i].
    let mut ystart = ctx.load_constant(F::from((N - 1) as u64));
    let mut found = ctx.load_constant(F::ZERO);
    for i in 0..N {
        let i_const = Constant(F::from(i as u64));
        let not_found = gate.not(ctx, found);
        let take = gate.mul(ctx, not_found, row_has[i]);
        ystart = gate.select(ctx, i_const, ystart, take);
        let or_v = gate.add(ctx, found, row_has[i]);
        let and_v = gate.mul(ctx, found, row_has[i]);
        found = gate.sub(ctx, or_v, and_v);
    }
    let mut ymax = ctx.load_constant(F::ZERO);
    for i in 0..N {
        let i_const = Constant(F::from(i as u64));
        ymax = gate.select(ctx, i_const, ymax, row_has[i]);
    }
    let mut xstart = ctx.load_constant(F::from((N - 1) as u64));
    let mut cfound = ctx.load_constant(F::ZERO);
    for j in 0..N {
        let j_const = Constant(F::from(j as u64));
        let not_found = gate.not(ctx, cfound);
        let take = gate.mul(ctx, not_found, col_has[j]);
        xstart = gate.select(ctx, j_const, xstart, take);
        let or_v = gate.add(ctx, cfound, col_has[j]);
        let and_v = gate.mul(ctx, cfound, col_has[j]);
        cfound = gate.sub(ctx, or_v, and_v);
    }
    let mut xmax = ctx.load_constant(F::ZERO);
    for j in 0..N {
        let j_const = Constant(F::from(j as u64));
        xmax = gate.select(ctx, j_const, xmax, col_has[j]);
    }

    let one = ctx.load_constant(F::ONE);
    let ystop = gate.add(ctx, ymax, one);
    let xstop = gate.add(ctx, xmax, one);

    // Output cells: keep im[i][j] when ystart <= i < ystop && xstart <= j < xstop.
    for i in 0..N {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let i_p1 = gate.add(ctx, i_cell, one);
        let ge_yst = rc.is_less_than(ctx, ystart, i_p1, 32);
        let lt_yst = rc.is_less_than(ctx, i_cell, ystop, 32);
        let in_row = gate.and(ctx, ge_yst, lt_yst);
        for j in 0..N {
            let j_cell = ctx.load_constant(F::from(j as u64));
            let j_p1 = gate.add(ctx, j_cell, one);
            let ge_xst = rc.is_less_than(ctx, xstart, j_p1, 32);
            let lt_xst = rc.is_less_than(ctx, j_cell, xstop, 32);
            let in_col = gate.and(ctx, ge_xst, lt_xst);
            let in_box = gate.and(ctx, in_row, in_col);
            let out = gate.select(ctx, cells[i * N + j], zero, in_box);
            make_public.push(out);
        }
    }

    let height = gate.sub(ctx, ystop, ystart);
    let width = gate.sub(ctx, xstop, xstart);
    make_public.push(ystart);
    make_public.push(ystop);
    make_public.push(xstart);
    make_public.push(xstop);
    make_public.push(height);
    make_public.push(width);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(argwhere_bbox_10x10, args);
}
