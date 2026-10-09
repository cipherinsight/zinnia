// pythran_tax__tax: literal translation of kernel.py (LITERAL_BASELINES.md).
// CHECKER BLOCKER: Halo2 L5/L6 per-operation input bounds conflict with the checker's
// zero redundant_input_checks requirement; see LITERAL_NPB3.md. No diagnostic bypass.
// Each mask compacts by running count and equality tests per input/output pair; padding effects are gated after array operations.
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

    let d = q.load_floats(ctx, &d);
    // Source mask and compacted segment 1.
    let lower: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[0] as f64));
            q.gt(ctx, x, y)
        })
        .collect();
    let keep = lower;
    let (s0, active0) = compact_scan(&q, ctx, &d, &keep);
    let product: Vec<Cell> = s0
        .iter()
        .map(|v| q.mul(ctx, *v, Constant(fe(quantize(BASE[0])))).cell)
        .collect();
    let seg0: Vec<Cell> = (0..N)
        .map(|p| {
            let v = q.add(ctx, product[p], Constant(fe(quantize(CONSTANT[0]))));
            q.gate.mul(ctx, v, active0[p])
        })
        .collect();
    // Source mask and compacted segment 2.
    let lower: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[1] as f64));
            q.gt(ctx, x, y)
        })
        .collect();
    let upper: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[0] as f64));
            q.le(ctx, x, y)
        })
        .collect();
    let keep: Vec<Cell> = (0..N)
        .map(|p| q.gate.and(ctx, lower[p], upper[p]))
        .collect();
    let (s1, active1) = compact_scan(&q, ctx, &d, &keep);
    let product: Vec<Cell> = s1
        .iter()
        .map(|v| q.mul(ctx, *v, Constant(fe(quantize(BASE[1])))).cell)
        .collect();
    let seg1: Vec<Cell> = (0..N)
        .map(|p| {
            let v = q.add(ctx, product[p], Constant(fe(quantize(CONSTANT[1]))));
            q.gate.mul(ctx, v, active1[p])
        })
        .collect();
    // Source mask and compacted segment 3.
    let lower: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[2] as f64));
            q.gt(ctx, x, y)
        })
        .collect();
    let upper: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[1] as f64));
            q.le(ctx, x, y)
        })
        .collect();
    let keep: Vec<Cell> = (0..N)
        .map(|p| q.gate.and(ctx, lower[p], upper[p]))
        .collect();
    let (s2, active2) = compact_scan(&q, ctx, &d, &keep);
    let seg2: Vec<Cell> = (0..N)
        .map(|p| {
            let v = q.sub(ctx, s2[p], Constant(fe(quantize(OFFSET[2]))));
            q.gate.mul(ctx, v, active2[p])
        })
        .collect();
    // Source mask and compacted segment 4.
    let lower: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[3] as f64));
            q.gt(ctx, x, y)
        })
        .collect();
    let upper: Vec<Cell> = d
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            let y = q.bounded_const(ctx, quantize(THRESHOLDS[2] as f64));
            q.le(ctx, x, y)
        })
        .collect();
    let keep: Vec<Cell> = (0..N)
        .map(|p| q.gate.and(ctx, lower[p], upper[p]))
        .collect();
    let (s3, active3) = compact_scan(&q, ctx, &d, &keep);
    let seg3: Vec<Cell> = (0..N)
        .map(|p| {
            let v = q.sub(ctx, s3[p], Constant(fe(quantize(OFFSET[3]))));
            q.gate.mul(ctx, v, active3[p])
        })
        .collect();
    let product: Vec<Cell> = seg2
        .iter()
        .map(|v| q.mul(ctx, *v, Constant(fe(quantize(SLOPE[2])))).cell)
        .collect();
    let prog2: Vec<Cell> = (0..N)
        .map(|p| {
            let v = q.add(ctx, product[p], Constant(fe(quantize(BASE[2]))));
            q.gate.mul(ctx, v, active2[p])
        })
        .collect();
    let product: Vec<Cell> = seg3
        .iter()
        .map(|v| q.mul(ctx, *v, Constant(fe(quantize(SLOPE[3])))).cell)
        .collect();
    let prog3: Vec<Cell> = (0..N)
        .map(|p| {
            let v = q.add(ctx, product[p], Constant(fe(quantize(BASE[3]))));
            q.gate.mul(ctx, v, active3[p])
        })
        .collect();
    let terms = &seg0;
    let mut total0 = ctx.load_zero();
    for p in 0..N {
        total0 = q.add(ctx, total0, terms[p]);
    }
    let terms = &seg1;
    let mut total1 = ctx.load_zero();
    for p in 0..N {
        total1 = q.add(ctx, total1, terms[p]);
    }
    let subtotal1 = q.add(ctx, total0, total1);
    let mut terms: Vec<Cell> = (0..N).map(|p| q.mul(ctx, seg2[p], prog2[p]).cell).collect();
    terms = (0..N)
        .map(|p| {
            let v = q.add(ctx, terms[p], Constant(fe(quantize(CONSTANT[2]))));
            q.gate.mul(ctx, v, active2[p])
        })
        .collect();
    let mut total2 = ctx.load_zero();
    for p in 0..N {
        total2 = q.add(ctx, total2, terms[p]);
    }
    let subtotal2 = q.add(ctx, subtotal1, total2);
    let mut terms: Vec<Cell> = (0..N).map(|p| q.mul(ctx, seg3[p], prog3[p]).cell).collect();
    let mut total3 = ctx.load_zero();
    for p in 0..N {
        total3 = q.add(ctx, total3, terms[p]);
    }
    let subtotal3 = q.add(ctx, subtotal2, total3);
    let mut income = ctx.load_zero();
    for v in d {
        income = q.add(ctx, income, v);
    }
    let result = q.canonical_div_nonzero(ctx, subtotal3, income).cell;
    out.float(result);
    Ok(())
}
fn compact_scan(q: &Q32, ctx: &mut Ctx, x: &[Cell], keep: &[Cell]) -> (Vec<Cell>, Vec<Cell>) {
    let mut count = ctx.load_zero();
    let mut positions = Vec::new();
    for i in 0..N {
        positions.push(count);
        count = q.add(ctx, count, keep[i]);
    }
    let mut selected = Vec::new();
    let mut active = Vec::new();
    for j in 0..N {
        let mut value = ctx.load_zero();
        let mut present = ctx.load_zero();
        for i in 0..N {
            let eq = q.eq(ctx, positions[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, keep[i], eq);
            let term = q.gate.mul(ctx, hit, x[i]);
            value = q.add(ctx, value, term);
            present = q.add(ctx, present, hit);
        }
        selected.push(value);
        active.push(present);
    }
    (selected, active)
}
