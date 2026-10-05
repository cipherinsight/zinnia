// DS-1000 case 507 — Halo2 baseline.
// Kernel:
//     mask = im == 0
//     rows = np.flatnonzero((~mask).sum(axis=1))
//     cols = np.flatnonzero((~mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
// im: Integer[5, 6]. Bounding box of nonzero entries.
//
// The kernel returns only the cropped array. Public output encoding is not
// specified by kernel.py, so align this handwritten baseline with the current
// Zinnia convention for these bbox cases: expose the fixed padded 5x6 payload
// with the crop packed into the top-left corner and zeros elsewhere, omitting
// separate shape metadata.

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

    let height_span = gate.sub(ctx, rmax, rmin);
    let one_h = ctx.load_constant(F::ONE);
    let height = gate.add(ctx, height_span, one_h);
    let width_span = gate.sub(ctx, cmax, cmin);
    let one_w = ctx.load_constant(F::ONE);
    let width = gate.add(ctx, width_span, one_w);
    let row_consts: Vec<_> = (0..R)
        .map(|i| ctx.load_constant(F::from(i as u64)))
        .collect();
    let col_consts: Vec<_> = (0..C)
        .map(|j| ctx.load_constant(F::from(j as u64)))
        .collect();
    for out_i in 0..R {
        let out_i_cell = row_consts[out_i];
        let row_active = rc.is_less_than(ctx, out_i_cell, height, 32);
        let src_row = gate.add(ctx, rmin, out_i_cell);
        let mut packed_row = Vec::with_capacity(C);
        for j in 0..C {
            let mut selected = cells[j];
            for i in 1..R {
                let eq = gate.is_equal(ctx, src_row, row_consts[i]);
                selected = gate.select(ctx, cells[i * C + j], selected, eq);
            }
            packed_row.push(selected);
        }
        for out_j in 0..C {
            let out_j_cell = col_consts[out_j];
            let col_active = rc.is_less_than(ctx, out_j_cell, width, 32);
            let src_col = gate.add(ctx, cmin, out_j_cell);
            let mut selected = packed_row[0];
            for j in 1..C {
                let eq = gate.is_equal(ctx, src_col, col_consts[j]);
                selected = gate.select(ctx, packed_row[j], selected, eq);
            }
            let active = gate.and(ctx, row_active, col_active);
            let out = gate.select(ctx, selected, zero, active);
            make_public.push(out);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(nonzero_bbox_5x6, args);
}
