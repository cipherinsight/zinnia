//! DS-1000 case 0500: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Reshape changes only the static shape; row-major wires keep their order.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[54])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    // a = a.reshape(-1, 3)
    let a = a;
    out.floats(&a);
    Ok(())
}
