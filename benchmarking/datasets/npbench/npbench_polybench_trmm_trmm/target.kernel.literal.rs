// npbench_polybench_trmm_trmm__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Preserve in-place loop order, then scale the complete B; an empty dot is constant zero.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 13; // = A_0
const NJ: usize = 16; // = B_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let a = inputs.floats("A", &[NI, NI])?;
    let b = inputs.floats("B", &[NI, NJ])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    // column[i] = A[i + 1:, i]
    let columns: Vec<Vec<Cell>> =
        (0..NI).map(|i| (i + 1..NI).map(|k| q.load_float(ctx, a[k * NI + i])).collect()).collect();
    let b = q.load_floats(ctx, &b);
    let mut b = b;
    for i in 0..NI { for j in 0..NJ {
        let mut raw = ctx.load_zero();
        for k in i+1..NI { let term = q.gate.mul(ctx,columns[i][k-i-1],b[k*NJ+j]); raw=q.add(ctx,raw,term); }
        let dot = if i+1 == NI { ctx.load_zero() } else { q.rescale(ctx,raw).cell };
        b[i*NJ+j] = q.add(ctx,b[i*NJ+j],dot);
    } }
    let result: Vec<Cell> = b.iter().map(|v| q.mul(ctx,*v,alpha).cell).collect();
    out.floats(&result); Ok(())
}
