// npbench_polybench_symm_symm__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Keep temp2 and the row expression after the j loop; recompute alpha*B for each source occurrence.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 10; // = C_0
const N: usize = 13; // = C_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[M, N])?;
    let a = inputs.floats("A", &[M, M])?;
    let b = inputs.floats("B", &[M, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let c = q.load_floats(ctx, &c);
    // a_low[i] = A[i, :i + 1]
    let a_low: Vec<Vec<Cell>> = (0..M).map(|i| (0..=i).map(|k| q.load_float(ctx, a[i * M + k])).collect()).collect();
    let b = q.load_floats(ctx, &b);

    let mut temp2 = vec![ctx.load_zero(); N];
    let mut c: Vec<Cell> = c.iter().map(|v| q.mul(ctx,*v,beta).cell).collect();
    for i in 0..M {
        for j in 0..N {
            let ab = q.mul(ctx,alpha,b[i*N+j]).cell;
            let terms: Vec<Cell> = (0..i).map(|k| q.mul(ctx,ab,a_low[i][k]).cell).collect();
            for k in 0..i { c[k*N+j] = q.add(ctx,c[k*N+j],terms[k]); }
            let mut raw = ctx.load_zero();
            for k in 0..i { let t=q.gate.mul(ctx,b[k*N+j],a_low[i][k]); raw=q.add(ctx,raw,t); }
            temp2[j] = if i == 0 { ctx.load_zero() } else { q.rescale(ctx,raw).cell };
        }
        let ab: Vec<Cell> = (0..N).map(|j| q.mul(ctx,alpha,b[i*N+j]).cell).collect();
        let diagonal: Vec<Cell> = (0..N).map(|j| q.mul(ctx,ab[j],a_low[i][i]).cell).collect();
        let at: Vec<Cell> = (0..N).map(|j| q.mul(ctx,alpha,temp2[j]).cell).collect();
        let update: Vec<Cell> = (0..N).map(|j| q.add(ctx,diagonal[j],at[j])).collect();
        for j in 0..N { c[i*N+j] = q.add(ctx,c[i*N+j],update[j]); }
    }
    out.floats(&c); Ok(())
}
