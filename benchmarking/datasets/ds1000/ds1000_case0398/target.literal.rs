// DS-1000 case 0398, literal translation of kernel.py (LITERAL_BASELINES.md).
// Loop-index branches fold at compile time; integer times Q32 stays exact, with separate products and additions.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.floats("A", &[30])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let values = q.load_floats(ctx, &values);
    let a = q.load_int(ctx, inputs.int("a")?);
    let b = q.load_int(ctx, inputs.int("b")?);
    let mut result = Vec::with_capacity(30);
    for k in 0..30 {
        let mut value = q.mul_int(ctx, values[k], a);
        if k > 0 {
            let term = q.mul_int(ctx, result[k - 1], b);
            value = q.add(ctx, value, term);
        }
        result.push(value);
    }
    out.floats(&result);
    Ok(())
}
