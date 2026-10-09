//! DS-1000 case 0438: literal translation of kernel.py (LITERAL_BASELINES.md).
//! corrcoef forms every entry of the full matrix, with every product and sum
//! in index order. Pearson uses the optimized exact integer centred moments,
//! one root and one quotient per entry, rather than rounded sample covariance.
//! Diagonal roots are computed in integer units (their radicands are perfect
//! squares), avoiding an extra Q32 root-window restriction absent from optimized.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let pv = inputs.ints("post", &[4])?;
    let dv = inputs.ints("distance", &[4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let data = [q.load_ints(ctx, &pv), q.load_ints(ctx, &dv)];
    let mut sums = Vec::new();
    for row in &data {
        let mut sum = q.int_const(ctx, 0);
        for x in row { sum = q.add(ctx, sum, *x); }
        sums.push(sum);
    }
    // Full centred-moment matrix: n sum(x*y) - sum(x) sum(y).
    let mut moments = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            let mut products = Vec::new();
            for k in 0..4 { products.push(q.gate.mul(ctx, data[i][k], data[j][k])); }
            let mut sum = q.int_const(ctx, 0);
            for product in products { sum = q.add(ctx, sum, product); }
            let scaled = q.gate.mul(ctx, sum, Constant(fe(4)));
            let correction = q.gate.mul(ctx, sums[i], sums[j]);
            moments.push(q.sub(ctx, scaled, correction));
        }
    }
    let diagonal = [moments[0], moments[3]];
    for d in diagonal { q.range.range_check(ctx, d, 64); }
    let mut correlation = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            let product = q.gate.mul(ctx, diagonal[i], diagonal[j]);
            let (numerator, denominator) = if i == j {
                // sqrt(d*d) is exactly d; integer units retain that exact root.
                (moments[2*i+j], q.isqrt(ctx, product).cell)
            } else {
                let radicand = q.int_to_float(ctx, product);
                let root = q.sqrt(ctx, radicand).cell;
                (q.int_to_float(ctx, moments[2*i+j]), root)
            };
            let denominator = q.bound(ctx, denominator);
            correlation.push(q.div_pos(ctx, numerator, denominator).cell);
        }
    }
    // corrcoef clips the complete matrix, including the unreturned entries.
    for value in &mut correlation {
        let v = q.bound(ctx, *value);
        let low = q.float_const(ctx, -1.0);
        let low = q.bound(ctx, low);
        let clipped = q.max(ctx, v, low).cell;
        let clipped = q.bound(ctx, clipped);
        let high = q.float_const(ctx, 1.0);
        let high = q.bound(ctx, high);
        *value = q.min(ctx, clipped, high).cell;
    }
    out.float(correlation[1]);
    Ok(())
}
