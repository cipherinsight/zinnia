//! DS-1000 case 0327: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Constant padding creates all 17 entries; seven constant outputs are omitted from instances.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.floats("A", &[10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &input);
    let length = 17;
    let mut result = a.to_vec();
    for _ in a.len()..length { result.push(q.float_const(ctx, 0.0)); }
    out.floats(&result[..10]);
    out.const_floats(&[0.0; 7]);
    Ok(())
}
