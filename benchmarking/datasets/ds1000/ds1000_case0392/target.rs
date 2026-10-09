//! DS-1000 case 0392, handwritten baseline.
// Constant row slice a[4:7, :] reads only 30 float elements; check each once.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[20, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let low = 4;
    let high = 7;
    let result = q.load_floats(ctx, &a[low * 10..high * 10]);
    out.floats(&result);
    Ok(())
}
