//! DS-1000 case 0303: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Only slice elements are read and domain-checked; discarded cells are not read.
//! Reshape/copy is a complete array of wire aliases, with no arithmetic.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[23])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let ncol = 5;
    let col = (a.len() / ncol) * ncol;
    let b = a[..col].to_vec();
    let b = q.load_floats(ctx, &b);
    let b: Vec<Vec<Cell>> = b.chunks(ncol).map(|row| row.to_vec()).collect();
    out.floats(&b.concat());
    Ok(())
}
