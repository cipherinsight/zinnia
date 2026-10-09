// pythran_multitype__check: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute x+x in full, then select the tuple branch. Boolean array truth uses its sole element.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.check.json.
const N: usize = 1; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let y = inputs.int("y")?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let y = q.load_int(ctx, y); let zero = q.gate.is_zero(ctx, y); let truthy = q.gate.not(ctx, zero);
    let result: Vec<Cell> = x.iter().map(|v| {
        let twice = q.add(ctx, *v, *v);
        q.gate.select(ctx, *v, twice, truthy)
    }).collect();
    out.floats(&result); out.int(y);
    Ok(())
}
