//! pythran_log_likelihood__log_likelihood (Pythran log_likelihood), halo2-base baseline.
//! Kernel: on data: Float[N], mean, sigma: Float, with the precondition
//! `sigma != 0`:
//!
//! ```text
//! s = (data - mean) ** 2 / (2 * sigma ** 2)
//! pdfs = numpy.exp(-s)
//! pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
//! return numpy.log(pdfs).sum()
//! ```
//!
//! Stable algebraic evaluation: log(exp(-s) / scale) = -s - log(scale).
//! Hoist log(sqrt(2 pi) * sigma) out of the reduction. This avoids losing
//! finite log densities when exp(-s) is smaller than one Q32 unit, and saves
//! one exp, log and division per element. Squares and the division defining
//! s retain the kernel's Q32 rounding; the identity is applied before any
//! Q32 materialization of the pdf. Positive sigma is necessary for finite
//! NumPy log densities and is enforced by log(scale).

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
    let normalization = q.log(ctx, scale);
    let terms: Vec<Cell> = data
        .iter()
        .map(|x| {
            let deviation = q.sub(ctx, *x, mean);
            let square = q.mul(ctx, deviation, deviation);
            let s = q.div_pos(ctx, square.cell, twice);
            q.add(ctx, s.cell, normalization.cell)
        })
        .collect();
    let total = q.sum(ctx, terms);
    let total = q.neg(ctx, total);
    out.float(total);
    Ok(())
}
