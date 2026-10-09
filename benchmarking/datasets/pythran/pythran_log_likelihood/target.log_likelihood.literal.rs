//! pythran_log_likelihood__log_likelihood: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize each array; sqrt(2*pi) is a Python-level constant.
//! Dynamic divisors use shared canonical_nonzero, never reused divisor proofs.

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
    let nonzero = q.gate.is_zero(ctx, sigma);
    q.gate.assert_is_const(ctx, &nonzero, &Fr::ZERO);
    let deviations: Vec<Cell> = data.iter().map(|v| q.sub(ctx,*v,mean)).collect();
    let squares: Vec<Cell> = deviations.iter().map(|v| q.pow_const(ctx,*v,2)).collect();
    let sigma_sq=q.pow_const(ctx,sigma,2);
    let twice=q.mul_int(ctx,Constant(fe(2)),sigma_sq);
    let s: Vec<Cell> = squares.iter().map(|v| q.canonical_div_nonzero(ctx,*v,twice).cell).collect();
    let negative: Vec<Cell> = s.iter().map(|v| q.neg(ctx,*v)).collect();
    let pdfs: Vec<Cell> = negative.iter().map(|v| { let v=q.bound(ctx,*v); q.canonical_exp(ctx,v).cell }).collect();
    let scale=q.mul(ctx,Constant(fe(quantize((2.0*std::f64::consts::PI).sqrt()))),sigma).cell;
    let pdfs: Vec<Cell> = pdfs.iter().map(|v| q.canonical_div_nonzero(ctx,*v,scale).cell).collect();
    let logs: Vec<Cell> = pdfs.iter().map(|v| q.canonical_log(ctx,*v).cell).collect();
    let mut total=q.float_const(ctx,0.0);
    for v in logs { total=q.add(ctx,total,v); }
    out.float(total); Ok(())
}
