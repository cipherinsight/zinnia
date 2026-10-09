// pythran_rosen__rosen: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize each elementwise array before the next operation, then sum in index order.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let squares: Vec<Cell> = (0..N - 1).map(|i| q.pow_const(ctx, x[i], 2)).collect();
    let gaps: Vec<Cell> = (0..N - 1)
        .map(|i| q.sub(ctx, x[i + 1], squares[i]))
        .collect();
    let squared_gaps: Vec<Cell> = gaps.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let t0: Vec<Cell> = squared_gaps
        .iter()
        .map(|v| q.mul_int(ctx, *v, Constant(fe(100))))
        .collect();
    let rest: Vec<Cell> = (0..N - 1)
        .map(|i| q.sub(ctx, Constant(fe(SCALE)), x[i]))
        .collect();
    let t1: Vec<Cell> = rest.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let sum: Vec<Cell> = (0..N - 1).map(|i| q.add(ctx, t0[i], t1[i])).collect();
    let mut total = ctx.load_zero();
    for term in sum {
        total = q.add(ctx, total, term);
    }
    out.float(total);
    Ok(())
}
