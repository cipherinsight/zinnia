//! pythran_tax__tax (Pythran tax), halo2-base baseline.
//! Kernel: progressive income tax of `d: Float[N]`: four masked selections
//! by the constant thresholds 256303 > 54057 > 13769 > 8820, each mapped and
//! summed, the total divided by `np.sum(d)`.
//! Preserve four source masks and separate compacted arrays, then evaluate
//! each segment's own formula with its original Q32 multiplication boundaries.
//! Padding contributes zero and does not enter the sums. No segment hint or
//! selected-coefficient polynomial substitutes for the four source formulas.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = d_0

/// The kernel's four tax segments (its text, not a size).
const SEGMENTS: usize = 4;

/// The thresholds, highest first: segment `s` (1 to 4) holds the incomes in
/// `(THRESHOLDS[s - 1], THRESHOLDS[s - 2]]` (no upper bound for segment 1).
const THRESHOLDS: [i64; SEGMENTS] = [256303, 54057, 13769, 8820];
/// Per segment 1 to 4: the offset `t` subtracted from `d`, the slope `k`, the
/// base `b` and the constant `e` of the term `s * (s * k + b) + e`.
const OFFSET: [f64; SEGMENTS] = [0.0, 0.0, 13769.0, 8820.0];
const SLOPE: [f64; SEGMENTS] = [0.0, 0.0, 2.2376e-06, 1.00727e-05];
const BASE: [f64; SEGMENTS] = [0.45, 0.42, 0.2397, 0.14];
const CONSTANT: [f64; SEGMENTS] = [-16164.53, -8475.44, 939.57, 0.0];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let d = inputs.floats("d", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let d = q.load_floats_bounded(ctx, &d);
    let inputs: Vec<Cell> = d.iter().map(|x| x.cell).collect();
    let mut totals = Vec::with_capacity(SEGMENTS);
    for segment in 0..SEGMENTS {
        let keep: Vec<Cell> = d.iter().map(|x| {
            let above = q.gt_const(ctx, *x, quantize(THRESHOLDS[segment] as f64));
            if segment == 0 { above } else {
                let upper = q.gt_const(ctx, *x, quantize(THRESHOLDS[segment - 1] as f64));
                let below = q.gate.not(ctx, upper);
                q.gate.and(ctx, above, below)
            }
        }).collect();
        let (length, selected) = q.select_kept(ctx, &inputs, &keep);
        let length = q.bound(ctx, length);
        let mut terms = Vec::with_capacity(N);
        for (i, x) in selected.iter().enumerate() {
            let active = q.gt_const(ctx, length, i as i128);
            let shifted = if segment < 2 { *x } else {
                let offset = q.gate.mul(ctx, active, Constant(fe(quantize(OFFSET[segment]))));
                q.sub(ctx, *x, offset)
            };
            let product = if segment < 2 {
                q.mul(ctx, shifted, Constant(fe(quantize(BASE[segment])))).cell
            } else {
                let slope = q.mul(ctx, shifted, Constant(fe(quantize(SLOPE[segment])))).cell;
                let prog = q.add(ctx, slope, Constant(fe(quantize(BASE[segment]))));
                q.mul(ctx, shifted, prog).cell
            };
            let constant = q.gate.mul(ctx, active, Constant(fe(quantize(CONSTANT[segment]))));
            terms.push(q.add(ctx, product, constant));
        }
        totals.push(q.sum(ctx, terms));
    }
    let total = q.sum(ctx, totals);
    let income = q.sum(ctx, inputs);
    // Source division asserts a nonzero denominator but does not narrow
    // this sum of ten inputs to signed64 before division.
    let result = q.canonical_div_nonzero(ctx, total, income);
    out.float(result.cell);
    Ok(())
}
