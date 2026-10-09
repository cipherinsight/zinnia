//! pythran_multitype__check2 (Pythran multitype), halo2-base baseline.
//! Kernel: `(x, y) if y else (x + x, y)` on `x: Float[N]` and
//! `y: Boolean[N]` (N = 1: `if y` takes the truth value of a one-element
//! array); the tuple is output in order.
//! The branch is a select on the loaded bit; `y` is output as loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.check2.json.
const N: usize = 1; // = x_0
const _: () = assert!(N == 1, "`if y` needs a one-element array");

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let y = inputs.bools("y", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let y = q.load_bools(ctx, &y);
    let falsy = q.gate.not(ctx, y[0]);
    // `x` if `y` is true, else `x + x`: x + (1 - y) * x.
    let result: Vec<Cell> = x.iter().map(|v| q.gate.mul_add(ctx, falsy, *v, *v)).collect();
    out.floats(&result);
    out.bools(&y);
    Ok(())
}
