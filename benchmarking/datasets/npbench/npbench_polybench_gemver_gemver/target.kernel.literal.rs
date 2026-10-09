// npbench_polybench_gemver_gemver__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize both outer products and every intermediate array in expression order; dot uses one rescale.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = A_0
const M: usize = 16; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let a = inputs.floats("A", &[N, M])?;
    let rows = ["u1", "u2", "w", "y"].map(|name| inputs.floats(name, &[N]));
    let cols = ["v1", "v2", "x", "z"].map(|name| inputs.floats(name, &[M]));
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let a = q.load_floats(ctx, &a);
    let mut loaded = Vec::new();
    for v in rows.into_iter().chain(cols) {
        loaded.push(q.load_floats(ctx, &v?));
    }
    let [u1, u2, w, y, v1, v2, x, z] = &loaded[..] else {
        unreachable!()
    };

    let mut first = Vec::new();
    for i in 0..N {
        for j in 0..M {
            first.push(q.mul(ctx, u1[i], v1[j]).cell);
        }
    }
    let mut second = Vec::new();
    for i in 0..N {
        for j in 0..M {
            second.push(q.mul(ctx, u2[i], v2[j]).cell);
        }
    }
    let updates: Vec<Cell> = (0..N * M)
        .map(|i| q.add(ctx, first[i], second[i]))
        .collect();
    let a: Vec<Cell> = (0..N * M).map(|i| q.add(ctx, a[i], updates[i])).collect();
    let by: Vec<Cell> = (0..N).map(|i| q.mul(ctx, beta, y[i]).cell).collect();
    let mut product = Vec::new();
    for j in 0..M {
        let mut raw = ctx.load_zero();
        for i in 0..N {
            let t = q.gate.mul(ctx, by[i], a[i * M + j]);
            raw = q.add(ctx, raw, t);
        }
        product.push(q.rescale(ctx, raw).cell);
    }
    let update: Vec<Cell> = (0..M).map(|j| q.add(ctx, product[j], z[j])).collect();
    let x: Vec<Cell> = (0..M).map(|j| q.add(ctx, x[j], update[j])).collect();
    let aa: Vec<Cell> = (0..N * M).map(|i| q.mul(ctx, alpha, a[i]).cell).collect();
    let mut product = Vec::new();
    for i in 0..N {
        let mut raw = ctx.load_zero();
        for j in 0..M {
            let t = q.gate.mul(ctx, aa[i * M + j], x[j]);
            raw = q.add(ctx, raw, t);
        }
        product.push(q.rescale(ctx, raw).cell);
    }
    let w: Vec<Cell> = (0..N).map(|i| q.add(ctx, w[i], product[i])).collect();
    out.floats(&a);
    out.floats(&w);
    out.floats(&x);
    Ok(())
}
