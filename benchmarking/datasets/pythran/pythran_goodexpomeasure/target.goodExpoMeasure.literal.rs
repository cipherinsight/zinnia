// pythran_goodexpomeasure__goodExpoMeasure: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize each source array; round-to-even uses shared divmod without local hints; uint8 cast keeps optimized backend domains.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const C: usize = 16; // = inRGB_0
const H: usize = 16; // = inRGB_1
const W: usize = 16; // = inRGB_2

/// The planes the kernel reads (R, G, B), and the centre value, from its text.
const PLANES: usize = 3;
const _: () = assert!(C >= PLANES, "IndexError");

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let in_rgb = inputs.ints("inRGB", &[C, H, W])?;
    let sigma = inputs.float("sigma")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let planes = q.load_ints(ctx, &in_rgb[..PLANES * H * W]);
    let sigma = q.load_float(ctx, sigma);
    let r: Vec<Cell> = (0..H * W)
        .map(|p| q.int_to_float(ctx, planes[0 * H * W + p]))
        .collect();
    let g: Vec<Cell> = (0..H * W)
        .map(|p| q.int_to_float(ctx, planes[1 * H * W + p]))
        .collect();
    let b: Vec<Cell> = (0..H * W)
        .map(|p| q.int_to_float(ctx, planes[2 * H * W + p]))
        .collect();
    // goodExpoR = exp(-(R-128)**2/sigma)
    let delta: Vec<Cell> = r
        .iter()
        .map(|v| q.sub(ctx, *v, Constant(fe(128 * SCALE))))
        .collect();
    let square: Vec<Cell> = delta.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let negative: Vec<Cell> = square.iter().map(|v| q.neg(ctx, *v)).collect();
    let quotient: Vec<Cell> = negative
        .iter()
        .map(|v| q.canonical_div_nonzero(ctx, *v, sigma).cell)
        .collect();
    let er: Vec<Cell> = quotient
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            q.canonical_exp(ctx, x).cell
        })
        .collect();
    // goodExpoG = exp(-(G-128)**2/sigma)
    let delta: Vec<Cell> = g
        .iter()
        .map(|v| q.sub(ctx, *v, Constant(fe(128 * SCALE))))
        .collect();
    let square: Vec<Cell> = delta.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let negative: Vec<Cell> = square.iter().map(|v| q.neg(ctx, *v)).collect();
    let quotient: Vec<Cell> = negative
        .iter()
        .map(|v| q.canonical_div_nonzero(ctx, *v, sigma).cell)
        .collect();
    let eg: Vec<Cell> = quotient
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            q.canonical_exp(ctx, x).cell
        })
        .collect();
    // goodExpoB = exp(-(B-128)**2/sigma)
    let delta: Vec<Cell> = b
        .iter()
        .map(|v| q.sub(ctx, *v, Constant(fe(128 * SCALE))))
        .collect();
    let square: Vec<Cell> = delta.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let negative: Vec<Cell> = square.iter().map(|v| q.neg(ctx, *v)).collect();
    let quotient: Vec<Cell> = negative
        .iter()
        .map(|v| q.canonical_div_nonzero(ctx, *v, sigma).cell)
        .collect();
    let eb: Vec<Cell> = quotient
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            q.canonical_exp(ctx, x).cell
        })
        .collect();
    let rg: Vec<Cell> = (0..H * W).map(|p| q.mul(ctx, er[p], eg[p]).cell).collect();
    let good: Vec<Cell> = (0..H * W).map(|p| q.mul(ctx, rg[p], eb[p]).cell).collect();
    let rounded: Vec<Cell> = good.iter().map(|v| round_hundredths(&q, ctx, *v)).collect();
    let scaled: Vec<Cell> = rounded
        .iter()
        .map(|v| q.mul(ctx, *v, Constant(fe(255 * SCALE))).cell)
        .collect();
    let result: Vec<Cell> = scaled
        .iter()
        .map(|v| {
            let v = q.bound(ctx, *v);
            let integer = q.float_to_int(ctx, v).cell;
            q.floor_div_const(ctx, integer, 256).1
        })
        .collect();
    out.ints(&result);
    Ok(())
}
// Nearest even hundredths using only shared division and remainder gadgets.
fn round_hundredths(q: &Q32, ctx: &mut Ctx, x: Cell) -> Cell {
    let scaled = q.mul_int(ctx, x, Constant(fe(100)));
    let shifted = q.add(ctx, scaled, Constant(fe(SCALE / 2)));
    let (m, f) = q.floor_div_const(ctx, shifted, SCALE as u64);
    let odd = q.floor_div_const(ctx, m.cell, 2).1;
    let tie = q.eq(ctx, f, Constant(fe(0)));
    let down = q.gate.mul(ctx, tie, odd);
    let n = q.sub(ctx, m.cell, down);
    let float = q.int_to_float(ctx, n);
    q.div_const(ctx, float, 100.0).cell
}
