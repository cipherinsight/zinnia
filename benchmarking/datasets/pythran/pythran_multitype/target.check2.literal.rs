// pythran_multitype__check2: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute x+x in full, then select the tuple branch. Boolean array truth uses its sole element.

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
    let truthy = y[0];
    let twice: Vec<Cell> = x.iter().map(|v| q.add(ctx, *v, *v)).collect();
    let result: Vec<Cell> = (0..N)
        .map(|i| q.gate.select(ctx, x[i], twice[i], truthy))
        .collect();
    out.floats(&result);
    out.bools(&y);
    Ok(())
}
