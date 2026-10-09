//! npbench_polybench_gemver_gemver__kernel (NPBench polybench/gemver), halo2-base baseline.
//! Kernel: on A: Float[N, M], u1, u2, w, y: Float[N] and v1, v2, x, z:
//! Float[M], in place: `A += outer(u1, v1) + outer(u2, v2)`,
//! `x += beta * y @ A + z`, `w += alpha * A @ x`; the outputs are `A`, `w`,
//! `x` after the call (`mutated_outputs`, parameter order).
//! Rounding follows NumPy's evaluation order: each element of an outer
//! product, of `beta * y` and of `alpha * A` is a rounded product (NumPy
//! materializes them), each element of a matrix-vector product is one
//! reduction rescaled once (`q.dot`): `(beta * y) @ A` and `(alpha * A) @ x`.

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
    let [u1, u2, w, y, v1, v2, x, z] = &loaded[..] else { unreachable!() };

    // A += np.outer(u1, v1) + np.outer(u2, v2)
    let mut a_new = Vec::with_capacity(N * M);
    for i in 0..N {
        for j in 0..M {
            let p1 = q.mul(ctx, u1[i], v1[j]);
            let p2 = q.mul(ctx, u2[i], v2[j]);
            let outer = q.add(ctx, p1.cell, p2.cell);
            a_new.push(q.add(ctx, a[i * M + j], outer));
        }
    }
    // x += beta * y @ A + z, i.e. x + ((beta * y) @ A + z)
    let beta_y: Vec<Cell> = y.iter().map(|yi| q.mul(ctx, beta, *yi).cell).collect();
    let mut x_new = Vec::with_capacity(M);
    for j in 0..M {
        let column: Vec<Cell> = (0..N).map(|i| a_new[i * M + j]).collect();
        let product = q.dot(ctx, &beta_y, &column);
        let update = q.add(ctx, product.cell, z[j]);
        x_new.push(q.add(ctx, x[j], update));
    }
    // w += alpha * A @ x, i.e. w + (alpha * A) @ x
    let mut w_new = Vec::with_capacity(N);
    for (i, row) in a_new.chunks(M).enumerate() {
        let alpha_row: Vec<Cell> = row.iter().map(|aij| q.mul(ctx, alpha, *aij).cell).collect();
        let product = q.dot(ctx, &alpha_row, &x_new);
        w_new.push(q.add(ctx, w[i], product.cell));
    }
    out.floats(&a_new);
    out.floats(&w_new);
    out.floats(&x_new);
    Ok(())
}
