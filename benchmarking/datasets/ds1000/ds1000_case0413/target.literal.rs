//! DS-1000 case 0413: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Complex values are interleaved (real, imaginary) Q32 components, each checked.
//! Exact nonzero imaginary test; length counts complex items, padded unit is two.
//! Prefix-count equality scan computes every input/output pair.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.complexes("x", &[10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    // x.imag != 0: first materialize the complete projection and mask.
    let imag: Vec<Cell> = (0..10).map(|i| x[2 * i + 1]).collect();
    let keep: Vec<Cell> = imag.iter().map(|im| {
        let zero = q.eq(ctx, *im, Constant(Fr::ZERO));
        q.gate.not(ctx, zero)
    }).collect();
    // x[mask]: running count, then each output slot scans all input items.
    let mut prefix = vec![q.int_const(ctx, 0)];
    for i in 0..10 { prefix.push(q.add(ctx, prefix[i], keep[i])); }
    let mut result = Vec::with_capacity(20);
    for j in 0..10 {
        let mut real = q.int_const(ctx, 0);
        let mut imag = q.int_const(ctx, 0);
        for i in 0..10 {
            let slot = q.eq(ctx, prefix[i], Constant(fe(j as i128)));
            let active = q.gate.and(ctx, keep[i], slot);
            let re = q.gate.mul(ctx, active, x[2 * i]);
            let im = q.gate.mul(ctx, active, x[2 * i + 1]);
            real = q.add(ctx, real, re);
            imag = q.add(ctx, imag, im);
        }
        result.extend([real, imag]);
    }
    out.dynamic(prefix[10], &result, Kind::Float, 2);
    Ok(())
}
