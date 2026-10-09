// DS-1000 case 0391, literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant slicing reads only columns 4, 5 and 6; each read input is checked once.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[20, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let (low, high) = (4, 7);
    let mut result = Vec::new();
    for i in 0..20 {
        for j in low..high {
            result.push(q.load_float(ctx, raw[i * 10 + j]));
        }
    }
    out.floats(&result);
    Ok(())
}
