//! pythran_multitype__check (Pythran multitype), halo2-base baseline.
//! Kernel: `(x, y) if y else (x + x, y)` on `x: Float[N]` (N = 1) and a
//! private integer `y`; the tuple is output in order.
//! The branch is a select on `y == 0`; `y` is output as loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.check.json.
const N: usize = 1; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let y = inputs.int("y")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let y = q.load_int(ctx, y);
    let falsy = q.gate.is_zero(ctx, y);
    // `x` if `y` is truthy, else `x + x`: x + (y == 0) * x.
    let result: Vec<Cell> = x.iter().map(|v| q.gate.mul_add(ctx, falsy, *v, *v)).collect();
    out.floats(&result);
    out.int(y);
    Ok(())
}
