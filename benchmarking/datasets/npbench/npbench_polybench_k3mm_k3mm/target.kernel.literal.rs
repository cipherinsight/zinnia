// npbench_polybench_k3mm_k3mm__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Evaluate three matrix products left to right; each dot accumulates in index order with one rescale.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 13; // = A_0
const NJ: usize = 15; // = A_1
const NK: usize = 14; // = B_1
const NL: usize = 16; // = C_1
const NM: usize = 16; // = D_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[NI, NJ])?;
    let b = inputs.floats("B", &[NJ, NK])?;
    let c = inputs.floats("C", &[NK, NL])?;
    let d = inputs.floats("D", &[NL, NM])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let c = q.load_floats(ctx, &c);
    let d = q.load_floats(ctx, &d);
    let ab = matmul(&q,ctx,&a,&b,NI,NJ,NK);
    let abc = matmul(&q,ctx,&ab,&c,NI,NK,NL);
    let result = matmul(&q,ctx,&abc,&d,NI,NL,NM);
    out.floats(&result); Ok(())
}
fn matmul(q: &Q32, ctx: &mut Ctx, a: &[Cell], b: &[Cell], rows: usize, inner: usize, cols: usize) -> Vec<Cell> {
    let mut result = Vec::new();
    for i in 0..rows { for j in 0..cols {
        let mut raw = ctx.load_zero();
        for k in 0..inner { let term = q.gate.mul(ctx,a[i*inner+k],b[k*cols+j]); raw = q.add(ctx,raw,term); }
        result.push(q.rescale(ctx,raw).cell);
    } }
    result
}
