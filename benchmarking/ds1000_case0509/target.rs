// DS-1000 case 509 — Halo2 baseline.
// Kernel:
//     mask = im == 0
//     rows = np.flatnonzero((mask).sum(axis=1))
//     cols = np.flatnonzero((mask).sum(axis=0))
//     result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
// im: Float[10, 10]. Bounding box of rows / cols that contain at least one
// zero. Uses the EqualFIR pattern for the float-equality test.
//
// Output shape is data-dependent — expose all 100 cells masked-to-zero
// outside the bbox plus shape metadata.

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

    let one = ctx.load_constant(F::ONE);
    let rmax_p1 = gate.add(ctx, rmax, one);
    let cmax_p1 = gate.add(ctx, cmax, one);
    for i in 0..N {
        let i_cell = ctx.load_constant(F::from(i as u64));
        let i_p1 = gate.add(ctx, i_cell, one);
        let ge_rmin = rc.is_less_than(ctx, rmin, i_p1, 32);
        let lt_rmax_p1 = rc.is_less_than(ctx, i_cell, rmax_p1, 32);
        let in_row = gate.and(ctx, ge_rmin, lt_rmax_p1);
        for j in 0..N {
            let j_cell = ctx.load_constant(F::from(j as u64));
            let j_p1 = gate.add(ctx, j_cell, one);
            let ge_cmin = rc.is_less_than(ctx, cmin, j_p1, 32);
            let lt_cmax_p1 = rc.is_less_than(ctx, j_cell, cmax_p1, 32);
            let in_col = gate.and(ctx, ge_cmin, lt_cmax_p1);
            let in_box = gate.and(ctx, in_row, in_col);
            let out = gate.select(ctx, cells[i * N + j], zero_fp, in_box);
            make_public.push(out);
        }
    }

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
    run(zero_bbox_10x10, args);
}
