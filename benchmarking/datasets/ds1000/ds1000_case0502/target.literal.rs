//! DS-1000 case 0502: literal translation of kernel.py (LITERAL_BASELINES.md).
//! One standard indexed read per row; newaxis and the final slice are wiring.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw_a = inputs.floats("a", &[13, 13, 2])?;
    let raw_b = inputs.ints("b", &[13, 13])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw_a);
    let b = q.load_ints(ctx, &raw_b);
    // take_along_axis; newaxis and [..., 0] only change the shape.
    let mut arr = Vec::new();
    for i in 0..169 {
        arr.push(q.index(ctx, &a[i * 2..(i + 1) * 2], b[i]));
    }
    out.floats(&arr);
    Ok(())
}
