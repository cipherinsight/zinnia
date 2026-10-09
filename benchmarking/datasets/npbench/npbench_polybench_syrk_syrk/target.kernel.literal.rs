// npbench_polybench_syrk_syrk__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// The scalar alpha*A[i,k] is evaluated once at its source location, then broadcast; products precede the slice update.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 14; // = A_0
const M: usize = 10; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[N, N])?;
    let a = inputs.floats("A", &[N, M])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let mut c = q.load_floats(ctx, &c);
    let a = q.load_floats(ctx, &a);
    for i in 0..N {
        for j in 0..=i {
            c[i * N + j] = q.mul(ctx, c[i * N + j], beta).cell;
        }
        for k in 0..M {
            let scale = q.mul(ctx, alpha, a[i * M + k]).cell;
            let products: Vec<Cell> = (0..=i)
                .map(|j| q.mul(ctx, scale, a[j * M + k]).cell)
                .collect();
            for j in 0..=i {
                c[i * N + j] = q.add(ctx, c[i * N + j], products[j]);
            }
        }
    }
    out.floats(&c);
    Ok(())
}
