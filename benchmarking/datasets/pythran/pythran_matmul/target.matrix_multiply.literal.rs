// pythran_matmul__matrix_multiply: literal translation of kernel.py (LITERAL_BASELINES.md).
// zero(n,m) preserves the source shape (m rows, n columns); scalar products round before each addition.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = m0_0
const K: usize = 16; // = m0_1
const P: usize = 16; // = m1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m0 = inputs.floats("m0", &[N, K])?;
    let m1 = inputs.floats("m1", &[K, P])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let m0 = q.load_floats(ctx, &m0);
    let m1 = q.load_floats(ctx, &m1);
    let mut result = zero(&q, ctx, N, P);
    for i in 0..N {
        for j in 0..P {
            let mut r = q.float_const(ctx, 0.0);
            for k in 0..K {
                let product = q.mul(ctx, m0[i * K + k], m1[k * P + j]).cell;
                r = q.add(ctx, r, product);
            }
            result[i][j] = r;
        }
    }
    out.floats(&result.into_iter().flatten().collect::<Vec<_>>());
    Ok(())
}
fn zero(q: &Q32, ctx: &mut Ctx, n: usize, m: usize) -> Vec<Vec<Cell>> {
    (0..m)
        .map(|_| (0..n).map(|_| q.float_const(ctx, 0.0)).collect())
        .collect()
}
