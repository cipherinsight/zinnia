// pythran_sumarray3d__summation: literal translation of kernel.py (LITERAL_BASELINES.md).
// Nested compute is called in comprehension order; each j updates total then x,y,z accumulators. Scalar products round individually.
use crate::prelude::*;
const NW: usize = 16; // = weights_0
const NP: usize = 16; // = points_0
fn compute(
    q: &Q32,
    ctx: &mut Ctx,
    i: usize,
    pos: &[Cell],
    weights: &[Cell],
    points: &[Cell],
    sum_array3d: &mut [Cell],
) -> Cell {
    let pxi = points[i * 3];
    let pyi = points[i * 3 + 1];
    let pzi = points[i * 3 + 2];
    let mut total = q.float_const(ctx, 0.0);
    for j in 0..NW {
        let weight_j = weights[j];
        let _xj = pos[j * 3];
        let _yj = pos[j * 3 + 1];
        let _zj = pos[j * 3 + 2];
        let dx = q.sub(ctx, pxi, pos[j * 3]);
        let dy = q.sub(ctx, pyi, pos[j * 3 + 1]);
        let dz = q.sub(ctx, pzi, pos[j * 3 + 2]);
        let xx = q.mul(ctx, dx, dx).cell;
        let yy = q.mul(ctx, dy, dy).cell;
        let xy = q.add(ctx, xx, yy);
        let zz = q.mul(ctx, dz, dz).cell;
        let sum = q.add(ctx, xy, zz);
        q.range.range_check(ctx, sum, 63); // Optimized radicand domain.
        let root = q.sqrt(ctx, sum).cell;
        let den = q.bound(ctx, root);
        let dr = q.div(ctx, Constant(fe(SCALE)), den).cell;
        let p = q.mul(ctx, weight_j, dr).cell;
        total = q.add(ctx, total, p);
        let p = q.mul(ctx, weight_j, dx).cell;
        sum_array3d[i * 3] = q.add(ctx, sum_array3d[i * 3], p);
        let p = q.mul(ctx, weight_j, dy).cell;
        sum_array3d[i * 3 + 1] = q.add(ctx, sum_array3d[i * 3 + 1], p);
        let p = q.mul(ctx, weight_j, dz).cell;
        sum_array3d[i * 3 + 2] = q.add(ctx, sum_array3d[i * 3 + 2], p);
    }
    total
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let pos = inputs.floats("pos", &[NW, 3])?;
    let weights = inputs.floats("weights", &[NW])?;
    let points = inputs.floats("points", &[NP, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let pos = q.load_floats(ctx, &pos);
    let weights = q.load_floats(ctx, &weights);
    let points = q.load_floats(ctx, &points);
    let zero = q.float_const(ctx, 0.0);
    let mut sum_array3d = vec![zero; NP * 3];
    let sum_array: Vec<Cell> = (0..NP)
        .map(|i| compute(&q, ctx, i, &pos, &weights, &points, &mut sum_array3d))
        .collect();
    out.floats(&sum_array);
    out.floats(&sum_array3d);
    Ok(())
}
