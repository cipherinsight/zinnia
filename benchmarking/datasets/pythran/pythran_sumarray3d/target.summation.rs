//! pythran_sumarray3d__summation (Pythran sumarray3d), halo2-base baseline.
//! Kernel: on pos: Float[NW, 3], weights: Float[NW], points: Float[NP, 3],
//! for every point `i` and weight `j`, with `(dx, dy, dz) = points[i] - pos[j]`:
//! `dr = 1.0 / sqrt(dx * dx + dy * dy + dz * dz)`, `total += weights[j] * dr`
//! and `sum_array3d[i] += weights[j] * (dx, dy, dz)`; the output is
//! `(sum_array, sum_array3d)`, `sum_array[i]` the total of point `i`.
//! `dx * dx + dy * dy + dz * dz` is a sum of squares, one reduction rescaled
//! once (NPB_BASELINES.md section 5, rule 3: like the loops that accumulate
//! `tmp * tmp`, here written out); its root is the exact floor square root
//! and `1.0 / root` a float division by a positive divisor (a zero distance
//! has no proof: NumPy gives inf). `total` and each coordinate of
//! `sum_array3d[i]` accumulate products into one variable from 0, one
//! reduction each, rescaled once (rule 3).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NW: usize = 16; // = weights_0
const NP: usize = 16; // = points_0
/// The coordinates x, y, z: fixed by the kernel's text, not a size.
const AXES: usize = 3;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let pos = inputs.floats("pos", &[NW, AXES])?;
    let weights = inputs.floats("weights", &[NW])?;
    let points = inputs.floats("points", &[NP, AXES])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let pos = q.load_floats(ctx, &pos);
    let weights = q.load_floats(ctx, &weights);
    let points = q.load_floats(ctx, &points);
    let one = Constant(fe(SCALE));
    let mut sum_array = Vec::with_capacity(NP);
    let mut sum_array3d = Vec::with_capacity(NP * AXES);
    for point in points.chunks(AXES) {
        let mut dr = Vec::with_capacity(NW);
        let mut deltas: Vec<Vec<Cell>> = vec![Vec::with_capacity(NW); AXES];
        for j in 0..NW {
            let d: Vec<Cell> = (0..AXES).map(|a| q.sub(ctx, point[a], pos[j * AXES + a])).collect();
            let squares = q.dot(ctx, &d, &d);
            let root = q.sqrt(ctx, squares.cell);
            dr.push(q.div_pos(ctx, one, root).cell);
            for (a, delta) in d.into_iter().enumerate() {
                deltas[a].push(delta);
            }
        }
        sum_array.push(q.dot(ctx, &weights, &dr).cell);
        for delta in &deltas {
            sum_array3d.push(q.dot(ctx, &weights, delta).cell);
        }
    }
    out.floats(&sum_array);
    out.floats(&sum_array3d);
    Ok(())
}
