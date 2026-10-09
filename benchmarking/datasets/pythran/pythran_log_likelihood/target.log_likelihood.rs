//! Literal log-likelihood sequence: square, divide, exp, density divide,
//! log, sum. No log(exp(x)) identity or normalization-log hoisting.
//! Shared transcendental gadgets define canonical Q32 arithmetic.
use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = data_0
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[N])?;
    let mean = inputs.float("mean")?;
    let sigma = inputs.float("sigma")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let data = q.load_floats(ctx, &data);
    let mean = q.load_float(ctx, mean);
    let sigma = q.load_float(ctx, sigma);
    let sigma_sq = q.mul(ctx, sigma, sigma);
    let twice = q.add(ctx, sigma_sq.cell, sigma_sq.cell);
    let twice = q.bound(ctx, twice);
    let scale = q.mul(ctx, Constant(fe(quantize((2.0 * std::f64::consts::PI).sqrt()))), sigma);
    let mut terms = Vec::with_capacity(N);
    for x in data {
        let deviation = q.sub(ctx, x, mean);
        let square = q.mul(ctx, deviation, deviation);
        let s = q.div(ctx, square.cell, twice);
        let neg_s = q.neg(ctx, s.cell);
        let neg_s = q.bound(ctx, neg_s);
        let density = q.exp(ctx, neg_s);
        let normalized = q.div(ctx, density.cell, scale);
        terms.push(q.log(ctx, normalized).cell);
    }
    let result = q.sum(ctx, terms);
    out.float(result);
    Ok(())
}
