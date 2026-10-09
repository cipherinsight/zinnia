//! DS-1000 case 0392, literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant row slice a[4:7, :] reads only 30 float elements; check each once.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[20, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let low = 4;
    let high = 7;
    let mut result = Vec::new();
    for i in low..high {
        for j in 0..10 {
            result.push(q.load_float(ctx, a[i * 10 + j]));
        }
    }
    out.floats(&result);
    Ok(())
}
