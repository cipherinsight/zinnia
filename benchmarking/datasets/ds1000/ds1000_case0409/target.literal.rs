//! DS-1000 case 0409, halo2-base LITERAL baseline (LITERAL_BASELINES.md).
//! Literal translation of kernel.py:
//!   x_new = np.array(x)
//!   y_new = np.array(y)
//!   z = x_new + y_new
//! `np.array(...)` on an array is a copy (no-op on the values); `+` is
//! elementwise addition, one gadget call per element (L5). The kernel has no
//! library call, no control flow and no repeated subexpression, so there is
//! no room for a literal/optimized difference here: this file is the same as
//! the optimized baseline.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (x, y) = (inputs.floats("x", &[8, 6])?, inputs.floats("y", &[8, 6])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (x, y) = (q.load_floats(ctx, &x), q.load_floats(ctx, &y));
    let z: Vec<Cell> = x.iter().zip(&y).map(|(a, b)| q.add(ctx, *a, *b)).collect();
    out.floats(&z);
    Ok(())
}
