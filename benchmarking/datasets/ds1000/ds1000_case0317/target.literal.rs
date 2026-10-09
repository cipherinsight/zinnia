//! DS-1000 case 0317: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Reshape/copy is a complete array of wire aliases, with no arithmetic.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[10, 20, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    let result = a.to_vec();
    out.floats(&result);
    Ok(())
}
