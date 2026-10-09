// pythran_hasting__fweb: literal translation of kernel.py (LITERAL_BASELINES.md).
// Repeated predation expressions are recomputed at every occurrence. Unread t is omitted; canonical nonzero division matches optimized arithmetic.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const Y_LEN: usize = 3; // = y_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let y = inputs.floats("y", &[Y_LEN])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let y = q.load_floats(ctx, &y[..3]);
    let a1 = q.load_float(ctx, inputs.float("a1")?);
    let a2 = q.load_float(ctx, inputs.float("a2")?);
    let b1 = q.load_float(ctx, inputs.float("b1")?);
    let b2 = q.load_float(ctx, inputs.float("b2")?);
    let d1 = q.load_float(ctx, inputs.float("d1")?);
    let d2 = q.load_float(ctx, inputs.float("d2")?);
    let one = Constant(fe(SCALE));
    let factor = q.sub(ctx, one, y[0]);
    let growth = q.mul(ctx, y[0], factor).cell;
    let p0_a = q.mul(ctx, a1, y[0]).cell;
    let p0_n = q.mul(ctx, p0_a, y[1]).cell;
    let p0_b = q.mul(ctx, b1, y[0]).cell;
    let p0_d = q.add(ctx, one, p0_b);
    let p0 = q.canonical_div_nonzero(ctx, p0_n, p0_d).cell;
    let yp0 = q.sub(ctx, growth, p0);
    let p1_a = q.mul(ctx, a1, y[0]).cell;
    let p1_n = q.mul(ctx, p1_a, y[1]).cell;
    let p1_b = q.mul(ctx, b1, y[0]).cell;
    let p1_d = q.add(ctx, one, p1_b);
    let p1 = q.canonical_div_nonzero(ctx, p1_n, p1_d).cell;
    let p2_a = q.mul(ctx, a2, y[1]).cell;
    let p2_n = q.mul(ctx, p2_a, y[2]).cell;
    let p2_b = q.mul(ctx, b2, y[1]).cell;
    let p2_d = q.add(ctx, one, p2_b);
    let p2 = q.canonical_div_nonzero(ctx, p2_n, p2_d).cell;
    let difference = q.sub(ctx, p1, p2);
    let death1 = q.mul(ctx, d1, y[1]).cell;
    let yp1 = q.sub(ctx, difference, death1);
    let p3_a = q.mul(ctx, a2, y[1]).cell;
    let p3_n = q.mul(ctx, p3_a, y[2]).cell;
    let p3_b = q.mul(ctx, b2, y[1]).cell;
    let p3_d = q.add(ctx, one, p3_b);
    let p3 = q.canonical_div_nonzero(ctx, p3_n, p3_d).cell;
    let death2 = q.mul(ctx, d2, y[2]).cell;
    let yp2 = q.sub(ctx, p3, death2);
    out.floats(&[yp0, yp1, yp2]);
    Ok(())
}
