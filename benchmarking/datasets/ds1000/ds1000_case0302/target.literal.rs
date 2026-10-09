//! DS-1000 case 0302: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Reshape/copy is a complete array of wire aliases, with no arithmetic.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[20])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    let nrow = 5;
    let ncol = a.len() / nrow;
    let b: Vec<Vec<Cell>> = a.chunks(ncol).map(|row| row.to_vec()).collect();
    out.floats(&b.concat());
    Ok(())
}
