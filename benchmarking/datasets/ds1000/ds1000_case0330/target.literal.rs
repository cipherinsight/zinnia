//! DS-1000 case 0330: literal translation of kernel.py (LITERAL_BASELINES.md).
//! One shared pow_const call per element; repeated squaring is inside the power operation (L5).
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    let power = 6;
    let mut result = Vec::new();
    for i in 0..48 { result.push(q.pow_const(ctx, a[i], power)); }
    out.floats(&result);
    Ok(())
}
