// DS-1000 case 508 — Halo2 baseline.
// Kernel:
//     B = np.argwhere(A)
//     (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
//     result = A[ystart:ystop, xstart:xstop]
// A: Integer[10, 10]. Same bounding-box-of-nonzero structure as 0507, just
// expressed via np.argwhere + min/max.
//
// The kernel returns only the cropped array. Public output encoding is not
// specified by kernel.py, so align this handwritten baseline with the current
// Zinnia convention for these bbox cases: expose the fixed padded 10x10
// payload with the crop packed into the top-left corner and zeros elsewhere,
// omitting separate shape metadata.

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

    let height_span = gate.sub(ctx, ymax, ystart);
    let one_h = ctx.load_constant(F::ONE);
    let height = gate.add(ctx, height_span, one_h);
    let width_span = gate.sub(ctx, xmax, xstart);
    let one_w = ctx.load_constant(F::ONE);
    let width = gate.add(ctx, width_span, one_w);

    // Expose a fixed 10x10 payload with the dynamic crop packed into the
    // top-left corner, matching Zinnia's current padded-slice encoding.
    let row_consts: Vec<_> = (0..N)
        .map(|i| ctx.load_constant(F::from(i as u64)))
        .collect();
    let col_consts: Vec<_> = (0..N)
        .map(|j| ctx.load_constant(F::from(j as u64)))
        .collect();
    for out_i in 0..N {
        let out_i_cell = row_consts[out_i];
        let row_active = rc.is_less_than(ctx, out_i_cell, height, 32);
        let src_row = gate.add(ctx, ystart, out_i_cell);
        let mut packed_row = Vec::with_capacity(N);
        for j in 0..N {
            let mut selected = cells[j];
            for i in 1..N {
                let eq = gate.is_equal(ctx, src_row, row_consts[i]);
                selected = gate.select(ctx, cells[i * N + j], selected, eq);
            }
            packed_row.push(selected);
        }
        for out_j in 0..N {
            let out_j_cell = col_consts[out_j];
            let col_active = rc.is_less_than(ctx, out_j_cell, width, 32);
            let src_col = gate.add(ctx, xstart, out_j_cell);
            let mut selected = packed_row[0];
            for j in 1..N {
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
    run(argwhere_bbox_10x10, args);
}
