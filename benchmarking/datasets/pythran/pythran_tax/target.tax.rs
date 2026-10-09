//! pythran_tax__tax (Pythran tax), halo2-base baseline.
//! Kernel: progressive income tax of `d: Float[N]`: four masked selections
//! by the constant thresholds 256303 > 54057 > 13769 > 8820, each mapped and
//! summed, the total divided by `np.sum(d)`.
//! Only the sums are output, so each income contributes its own segment's
//! term to one sum (a masked element of the other segments contributes 0).
//! The four segment terms have one shape, `s * (s * k + b) + e` with
//! `s = d - t` (segments 1 and 2: `t = 0`, `k = 0`, `b` their rate), so
//! every income evaluates it once, with its segment's constants selected by
//! a one-hot of the segment: two rounded products per income, as NumPy
//! rounds them in its own segment (`floor(d * 0) = 0`, exact). Below the
//! lowest threshold every constant is 0 and the term is 0. The segment
//! one-hot is a hint pinned by two range checks (`segment`).

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
    let mut terms: Vec<Cell> = Vec::with_capacity(N);
    for x in &d {
        let hot = segment(&q, ctx, *x);
        let mut pick = |values: &[f64; SEGMENTS]| -> Cell {
            q.gate.inner_product(ctx, hot.iter().map(|h| Existing(*h)), values.iter().map(|v| Constant(fe(quantize(*v)))))
        };
        let (t, k, b, e) = (pick(&OFFSET), pick(&SLOPE), pick(&BASE), pick(&CONSTANT));
        let s = q.sub(ctx, x.cell, t);
        let slope = q.mul(ctx, s, k);
        let p = q.add(ctx, slope.cell, b);
        let product = q.mul(ctx, s, p);
        terms.push(q.add(ctx, product.cell, e));
    }
    let total = q.sum(ctx, terms);
    let income = q.sum(ctx, d.iter().map(|x| x.cell));
    let income = q.bound(ctx, income);
    let result = q.div(ctx, total, income);
    out.float(result.cell);
    Ok(())
}

/// The one-hot `[h1, h2, h3, h4]` of the segment of `x` (all zero below the
/// lowest threshold), as a hint pinned by two range checks: with `h_s` bits
/// summing to at most 1, the selected bounds `(lo, hi]` are one segment's
/// (the window's ends for the open sides), and `lo < x <= hi` holds for
/// exactly one segment, since they partition the window (`x` is in it).
/// Measured against four comparisons `x > THRESHOLDS[s]`
/// (3,938 / 671 advice / lookup cells against 4,458 / 911 for the circuit).
fn segment(q: &Q32, ctx: &mut Ctx, x: I64) -> [Cell; SEGMENTS] {
    let v = x.value();
    let s = THRESHOLDS.iter().position(|t| v > (*t as i128) * SCALE);
    let hot: Vec<Cell> = (0..SEGMENTS)
        .map(|i| {
            // Hint, pinned by the two range checks below.
            let h = ctx.load_witness(Fr::from((s == Some(i)) as u64));
            q.gate.assert_bit(ctx, h);
            h
        })
        .collect();
    let any = q.sum(ctx, hot.iter().copied());
    q.gate.assert_bit(ctx, any);
    // (lo, hi] per segment 0 to 4 as Q32 codes; segment 0 (no bit set) is
    // the base each selection adds its steps to.
    let window = 1i128 << 63;
    let code = |t: i64| t as i128 * SCALE;
    let mut bounds: Vec<(i128, i128)> = vec![(-window - 1, code(THRESHOLDS[SEGMENTS - 1]))];
    for (i, t) in THRESHOLDS.iter().enumerate() {
        let hi = if i == 0 { window - 1 } else { code(THRESHOLDS[i - 1]) };
        bounds.push((code(*t), hi));
    }
    let mut select = |side: fn(&(i128, i128)) -> i128| {
        let base = side(&bounds[0]);
        let steps = bounds[1..].iter().map(|b| Constant(fe(side(b) - base)));
        let picked = q.gate.inner_product(ctx, hot.iter().map(|h| Existing(*h)), steps);
        q.add(ctx, picked, Constant(fe(base)))
    };
    let lo = select(|b| b.0);
    let hi = select(|b| b.1);
    let above = q.sub(ctx, x.cell, lo);
    let above = q.sub(ctx, above, Constant(Fr::ONE));
    q.range.range_check(ctx, above, 64);
    let below = q.sub(ctx, hi, x.cell);
    q.range.range_check(ctx, below, 64);
    hot.try_into().expect("one bit per segment")
}
