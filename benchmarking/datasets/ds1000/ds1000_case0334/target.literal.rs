//! DS-1000 case 0334: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Form the stack; mean is a running sum in axis order, followed by shared float division by 3.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[50])?;
    let b = inputs.floats("b", &[50])?;
    let c = inputs.floats("c", &[50])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let arrays = [q.load_floats(ctx, &a), q.load_floats(ctx, &b), q.load_floats(ctx, &c)];
    let mut result = Vec::new();
    for i in 0..50 {
        let mut sum = arrays[0][i];
        for j in 1..3 { sum = q.add(ctx, sum, arrays[j][i]); }
        result.push(q.div_const(ctx, sum, 3.0).cell);
    }
    out.floats(&result);
    Ok(())
}
