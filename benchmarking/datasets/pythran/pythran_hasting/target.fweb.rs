//! pythran_hasting__fweb (Pythran hasting), halo2-base baseline.
//! Kernel: the right-hand side of the Hastings food-web ODE for `y: Float[3]`
//! and six scalar rates; `t` is never read (not loaded), nor are the elements
//! of `y` after the third (none at the fixture's shape).
//! Every product and quotient is rounded in Python's left-to-right order
//! (`a1 * y[0] * y[1] / (1.0 + b1 * y[0])` is `((a1 * y0) * y1) / (..)`).
//! The two predation terms `a1 y0 y1 / (1 + b1 y0)` and
//! `a2 y1 y2 / (1 + b2 y1)` appear twice each in the kernel with the same
//! operands, so each is computed once. The divisors can have either sign
//! (`q.div`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const Y_LEN: usize = 3; // = y_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let y = inputs.floats("y", &[Y_LEN])?;
    let rates = ["a1", "a2", "b1", "b2", "d1", "d2"].map(|name| inputs.float(name));
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let y = q.load_floats(ctx, &y[..3]);
    let mut loaded = Vec::with_capacity(rates.len());
    for rate in rates {
        loaded.push(q.load_float(ctx, rate?));
    }
    let [a1, a2, b1, b2, d1, d2] = loaded[..] else { unreachable!() };
    let one = Constant(fe(SCALE));

    // y0 * (1.0 - y0)
    let one_minus_y0 = q.sub(ctx, one, y[0]);
    let growth = q.mul(ctx, y[0], one_minus_y0);
    // a1 * y0 * y1 / (1.0 + b1 * y0) and a2 * y1 * y2 / (1.0 + b2 * y1)
    let predation1 = predation(&q, ctx, a1, y[0], y[1], b1);
    let predation2 = predation(&q, ctx, a2, y[1], y[2], b2);

    let yprime0 = q.sub(ctx, growth.cell, predation1);
    let yprime1 = q.sub(ctx, predation1, predation2);
    let death1 = q.mul(ctx, d1, y[1]);
    let yprime1 = q.sub(ctx, yprime1, death1.cell);
    let death2 = q.mul(ctx, d2, y[2]);
    let yprime2 = q.sub(ctx, predation2, death2.cell);
    out.floats(&[yprime0, yprime1, yprime2]);
    Ok(())
}

/// `a * u * v / (1.0 + b * u)`.
fn predation(q: &Q32, ctx: &mut Ctx, a: Cell, u: Cell, v: Cell, b: Cell) -> Cell {
    let au = q.mul(ctx, a, u);
    let auv = q.mul(ctx, au.cell, v);
    let bu = q.mul(ctx, b, u);
    let divisor = q.add(ctx, Constant(fe(SCALE)), bu.cell);
    let divisor = q.bound(ctx, divisor);
    q.div(ctx, auv.cell, divisor).cell
}
