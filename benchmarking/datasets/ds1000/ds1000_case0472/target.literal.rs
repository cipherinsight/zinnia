//! DS-1000 case 0472: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Static slice: only selected input cells are read and domain-checked (L7).
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[6, 2, 9, 3, 7, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    // result = a[-1:, ...]
    let values: Vec<f64> = (0..1512).map(|i| raw[7560 + i * 1]).collect();
    let result = q.load_floats(ctx, &values);
    out.floats(&result);
    Ok(())
}
