// DS-1000 case 509 — Halo2 baseline.
// Kernel:
//     mask = im == 0
//     rows = np.flatnonzero((mask).sum(axis=1))
//     cols = np.flatnonzero((mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
// im: Float[10, 10]. Bounding box of rows / cols that contain at least one
// zero. Uses the EqualFIR pattern for the float-equality test.
//
// The kernel returns only the cropped array. Public output encoding is not
// specified by kernel.py, so align this handwritten baseline with the current
// Zinnia convention for these bbox cases: expose the fixed padded 10x10
// payload with the crop packed into the top-left corner and zeros elsewhere,
// omitting separate shape metadata.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{AssignedValue, QuantumCell::Constant};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub im: Vec<f64>, // 10*10 = 100 floats, row-major
}

fn zero_bbox_10x10<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const N: usize = 10;
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let rc = builder.range_chip();
    let fpc = FixedPointChip::<F, P>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.im.len(), N * N);
    let cells: Vec<AssignedValue<F>> = input.im.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // mask[i][j] = (im[i][j] == 0.0) via EqualFIR (qsub + two range checks).
    let eps_pos = Constant(fpc.quantization(0.001));
    let eps_neg = Constant(fpc.quantization(-0.001));
    let zero_fp = ctx.load_witness(fpc.quantization(0.0));
    let mut mask = Vec::with_capacity(N * N);
    for k in 0..(N * N) {
        let diff = fpc.qsub(ctx, cells[k], zero_fp);
        let lt_p = rc.is_less_than(ctx, diff, eps_pos, 128);
        let gt_n = rc.is_less_than(ctx, eps_neg, diff, 128);
        let eq = gate.and(ctx, lt_p, gt_n);
        mask.push(eq);
    }

    let zero_i = ctx.load_constant(F::ZERO);
    let mut row_has = Vec::with_capacity(N);
    for i in 0..N {
        let mut s = mask[i * N];
        for j in 1..N { s = gate.add(ctx, s, mask[i * N + j]); }
        let is_z = gate.is_equal(ctx, s, zero_i);
        row_has.push(gate.not(ctx, is_z));
    }
    let mut col_has = Vec::with_capacity(N);
    for j in 0..N {
        let mut s = mask[j];
        for i in 1..N { s = gate.add(ctx, s, mask[i * N + j]); }
        let is_z = gate.is_equal(ctx, s, zero_i);
        col_has.push(gate.not(ctx, is_z));
    }

    let mut rmin = ctx.load_constant(F::from((N - 1) as u64));
    let mut found = ctx.load_constant(F::ZERO);
    for i in 0..N {
        let i_const = Constant(F::from(i as u64));
        let not_found = gate.not(ctx, found);
        let take = gate.mul(ctx, not_found, row_has[i]);
        rmin = gate.select(ctx, i_const, rmin, take);
        let or_v = gate.add(ctx, found, row_has[i]);
        let and_v = gate.mul(ctx, found, row_has[i]);
        found = gate.sub(ctx, or_v, and_v);
    }
    let mut rmax = ctx.load_constant(F::ZERO);
    for i in 0..N {
        let i_const = Constant(F::from(i as u64));
        rmax = gate.select(ctx, i_const, rmax, row_has[i]);
    }
    let mut cmin = ctx.load_constant(F::from((N - 1) as u64));
    let mut cfound = ctx.load_constant(F::ZERO);
    for j in 0..N {
        let j_const = Constant(F::from(j as u64));
        let not_found = gate.not(ctx, cfound);
        let take = gate.mul(ctx, not_found, col_has[j]);
        cmin = gate.select(ctx, j_const, cmin, take);
        let or_v = gate.add(ctx, cfound, col_has[j]);
        let and_v = gate.mul(ctx, cfound, col_has[j]);
        cfound = gate.sub(ctx, or_v, and_v);
    }
    let mut cmax = ctx.load_constant(F::ZERO);
    for j in 0..N {
        let j_const = Constant(F::from(j as u64));
        cmax = gate.select(ctx, j_const, cmax, col_has[j]);
    }

    let height_span = gate.sub(ctx, rmax, rmin);
    let one_h = ctx.load_constant(F::ONE);
    let height = gate.add(ctx, height_span, one_h);
    let width_span = gate.sub(ctx, cmax, cmin);
    let one_w = ctx.load_constant(F::ONE);
    let width = gate.add(ctx, width_span, one_w);
    let row_consts: Vec<_> = (0..N)
        .map(|i| ctx.load_constant(F::from(i as u64)))
        .collect();
    let col_consts: Vec<_> = (0..N)
        .map(|j| ctx.load_constant(F::from(j as u64)))
        .collect();
    for out_i in 0..N {
        let out_i_cell = row_consts[out_i];
        let row_active = rc.is_less_than(ctx, out_i_cell, height, 32);
        let src_row = gate.add(ctx, rmin, out_i_cell);
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
            let src_col = gate.add(ctx, cmin, out_j_cell);
            let mut selected = packed_row[0];
            for j in 1..N {
                let eq = gate.is_equal(ctx, src_col, col_consts[j]);
                selected = gate.select(ctx, packed_row[j], selected, eq);
            }
            let active = gate.and(ctx, row_active, col_active);
            let out = gate.select(ctx, selected, zero_fp, active);
            make_public.push(out);
        }
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(zero_bbox_10x10, args);
}
