// npbench_polybench_ludcmp_ludcmp__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Dots include empty reductions. Shared canonical nonzero division preserves the optimized arithmetic/domain contract for derived pivots.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let b = inputs.floats("b", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let zero = q.float_const(ctx, 0.0);
    let mut x = vec![zero; N];
    let mut y = vec![zero; N];
    for i in 0..N {
        for j in 0..i {
            let col: Vec<Cell> = (0..j).map(|k| a[k * N + j]).collect();
            let dot = q.dot(ctx, &a[i * N..i * N + j], &col).cell;
            a[i * N + j] = q.sub(ctx, a[i * N + j], dot);
            a[i * N + j] = q
                .canonical_div_nonzero(ctx, a[i * N + j], a[j * N + j])
                .cell;
        }
        for j in i..N {
            let col: Vec<Cell> = (0..i).map(|k| a[k * N + j]).collect();
            let dot = q.dot(ctx, &a[i * N..i * N + i], &col).cell;
            a[i * N + j] = q.sub(ctx, a[i * N + j], dot);
        }
    }
    for i in 0..N {
        let dot = q.dot(ctx, &a[i * N..i * N + i], &y[..i]).cell;
        y[i] = q.sub(ctx, b[i], dot);
    }
    for i in (0..N).rev() {
        let dot = q.dot(ctx, &a[i * N + i + 1..(i + 1) * N], &x[i + 1..]).cell;
        let numerator = q.sub(ctx, y[i], dot);
        x[i] = q.canonical_div_nonzero(ctx, numerator, a[i * N + i]).cell;
    }
    out.floats(&x);
    out.floats(&y);
    Ok(())
}
