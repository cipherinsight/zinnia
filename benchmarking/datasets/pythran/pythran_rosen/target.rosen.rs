//! pythran_rosen__rosen (Pythran rosen), halo2-base baseline.
//! Kernel: the Rosenbrock function of x: Float[N],
//! `np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)`.
//! Each `** 2` is a rounded product (an elementwise array NumPy materializes);
//! `100 * v` is exact; the sum is exact.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let one = Constant(fe(SCALE));
    let mut terms = Vec::with_capacity(2 * (N - 1));
    for i in 0..N - 1 {
        let square = q.mul(ctx, x[i], x[i]);
        let gap = q.sub(ctx, x[i + 1], square.cell);
        let gap_sq = q.mul(ctx, gap, gap);
        terms.push(q.gate.mul(ctx, gap_sq.cell, Constant(Fr::from(100))));
        let rest = q.sub(ctx, one, x[i]);
        terms.push(q.mul(ctx, rest, rest).cell);
    }
    let total = q.sum(ctx, terms);
    out.float(total);
    Ok(())
}
